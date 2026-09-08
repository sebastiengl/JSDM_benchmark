from ast import List
import torch
import torch.nn as nn
import pandas as p
from dataclasses import dataclass

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TabularEmbedder(nn.Module):
    def __init__(self, dropout_prob, input_size):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.null_embedding = None
        self.dropout_prob = dropout_prob
        self.input_size = input_size

        self.embed_size = 16
        self.bn = nn.BatchNorm1d(input_size)
        if use_cfg_embedding:
            self.null_embedding = nn.Parameter(torch.zeros(1, self.embed_size))
        self.ln = nn.LayerNorm(self.embed_size)
        self.encoder = nn.Linear(input_size, self.embed_size)
 
    def embed_drop(self, embedding, force_drop_ids=None):
        """
        Drops labels for classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(embedding.shape[0], device=embedding.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        drop_ids = drop_ids.unsqueeze(1)
        embedding = torch.where(drop_ids, self.null_embedding, embedding)
        return embedding

    def forward(self, x, train= False, force_drop_ids=None):
        x = x[0]
        x = self.bn(x)
        x = self.encoder(x)
        x = self.ln(x)
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            x = self.embed_drop(x, force_drop_ids)
        return x


class MarginalModel(nn.Module):
    def __init__(self, vocab, dropout_prob = 0.1, hidden_dims = [1000, 2056], env_emb_class = "TabularEmbedder", input_size = None):
        super().__init__()
        self.output_dim = vocab.nb_species
        self.dropout_prob = dropout_prob
        embedder_class = globals()[env_emb_class]
        self.env_embedder = embedder_class(0.0) if input_size is None else embedder_class(0.0, input_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        fc_layers = []
        last = self.env_embedder.embed_size
        for h in hidden_dims:
            fc_layers.append(nn.Linear(last, h))
            fc_layers.append(nn.LayerNorm(h))
            last = h

        fc_layers.append(nn.Linear(last, self.output_dim))
        self.proj = nn.Sequential(*fc_layers)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, y, tgt = None, survey_id = None):
        y = self.env_embedder(y)
        y = self.dropout(y)
        logits = self.proj(y)

        return logits

    def loss_func(self, lgt, tgt, _):
        return self.loss(lgt, tgt)



##########################


class AutoRegModel(nn.Module):
    def __init__(self, vocab, max_length, dropout_prob=0.1, depth=2, embed_dim= 512, pos_enc= False, env_emb_class = "TabularEmbedder", input_size = None):
        super().__init__()
        self.eos_token = vocab.eos_token 
        self.pad_token = vocab.pad_token
        self.vocab_size = vocab.size
        self.nb_species = vocab.nb_species
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob

        # Reserve positions for BOS and EOS
        self.max_length = max_length
        embedder_class = globals()[env_emb_class]
        self.env_embedder = embedder_class(0.0) if input_size is None else embedder_class(0.0, input_size)
        self.species_emb = nn.Embedding(self.vocab_size , self.embed_dim, padding_idx=self.eos_token) 
        self.pos_enc = pos_enc
        if self.pos_enc:
            self.pos_emb = nn.Embedding(self.max_length + 2, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        # CAT AND NOT ADD, TO FIX LATTER :
        input_dim = self.embed_dim + self.env_embedder.embed_size
        
        decoder_layer = nn.TransformerEncoderLayer(d_model= input_dim, nhead= 16, dropout= self.dropout_prob)  #np.floor(embed_dim / 32).astype(int), dropout=0.1)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers= depth)
        self.output_proj = nn.Linear(input_dim, self.nb_species + 1) # Only EOS_token
        self.dropout = nn.Dropout(self.dropout_prob)

        self.loss =  torch.nn.CrossEntropyLoss(ignore_index = self.pad_token)


    def forward(self, y, tgt, survey_id = None):

        B, L = tgt.shape
        context = self.env_embedder(y, train=self.training).unsqueeze(0)  # [1, B, E]
        tgt_emb = self.species_emb(tgt) # [B, L, E]

        if self.pos_enc:
            #tgt_emb = tgt_emb + pos_emb
            pos_ids = torch.arange(L, device=tgt.device)
            pos_emb = self.pos_emb(pos_ids)
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # [B, L, E]
            tgt_emb = pos_emb + tgt_emb


        bos = torch.zeros(1, B, self.embed_dim, device=tgt.device)  # [1, B, E]

        tgt_emb = tgt_emb.transpose(0, 1)  # [L, B, E]
        tgt_emb = torch.cat([bos, tgt_emb], dim=0)
        #tgt_emb = tgt_emb + context

        # CAT AND NO ADD, TO FIX LATER WITH DECODER EMD_DIM
        tgt_emb = torch.cat([tgt_emb, context.expand(L+1, -1, -1)], dim = -1)# Broadcast context to all positions

        tgt_emb = self.dropout(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L + 1).to(tgt.device)
        output = self.transformer_decoder(tgt_emb, tgt_mask)  # [L, B, E]
        logits = self.output_proj(output).transpose(0, 1)  # [B, L, V]
        return logits

    def loss_func(self, lgt, _, tokens):
        logits = lgt[:, :-1] # [B, L, V]
        logits_reshape = logits.reshape(-1, logits.size(-1)) # [B*L, V]
        target_reshape = tokens.reshape(-1) # [B*L]

        return self.loss(logits_reshape, target_reshape)




class CSVModel(nn.Module):
    def __init__(self, vocab, p_file):
        super().__init__()
        self.vocab_size = vocab.size
        self.nb_species = vocab.nb_species
        self.file = p.read_csv(p_file)


    def forward(self, y, _ = None, survey_id = None):
        if survey_id is None:
            raise ValueError("survey_id must be provided for this model.")
        row = self.file[self.file['survey_id'] == survey_id]
        
        logits = torch.tensor(row.iloc[0, 1:].values, dtype=torch.float32, device=y.device)
        return logits.unsqueeze(0)

        
    def loss_func(self, lgt, tgt, _):
        pass




@dataclass
class StochasticMixtureModel(nn.Module):
    sub_models : List

    def __post_init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        chosen = self.sub_models[int(torch.rand(1) * len(self.sub_models))]
        return chosen(*args, **kwargs)

