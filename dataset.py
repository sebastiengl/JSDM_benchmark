import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as p

class SpeciesVocab():
    def __init__(self, train_species, test_species):
        self.binary_matrix = False
        #test for different format 
        if "speciesId" not in train_species.columns :
            print("Assuming species data are in binary matrix format ...")
            self.binary_matrix  = True
            train_presence = train_species.iloc[:,1:].sum()
            test_presence = test_species.iloc[:,1:].sum()
            presence = (train_presence + test_presence > 0)

            all_species = np.arange(0,len(train_presence), dtype = int)[presence]
            self.species_counts = {i: int(train_presence.iloc[i]) for i in range(len(train_presence))}

        else :
            train_species = set(train_species["speciesId"].dropna().astype(int))
            test_species = set(test_species["speciesId"].dropna().astype(int))
            self.species_counts = train_species["speciesId"].value_counts().to_dict()
            all_species = sorted(train_species.union(test_species))

        for species_id in all_species:
            self.species_counts.setdefault(int(species_id), 0)

        self.vocab = {int(s): i for i, s in enumerate(all_species)}
        self.inv_vocab = {i: s for s, i in self.vocab.items()}
        self.nb_species = len(self.vocab)
        self.size = self.nb_species + 3

        self.eos_token = self.nb_species
        self.pad_token = self.nb_species + 1
        self.mask_token = self.nb_species + 2

        

    def get_index(self, species_id):
        try:
            if isinstance(species_id, np.ndarray):
                if species_id.size != 1:
                    raise ValueError(f"Species ID array {species_id} has multiple elements")
                species_id = species_id.item()
            species_id = int(species_id)
        except (TypeError, ValueError):
            raise ValueError(f"Species ID {species_id} is not a valid scalar species id")

        if species_id not in self.vocab:
            raise ValueError(f"Species ID {species_id} not found in vocab")
        return self.vocab[species_id]
    
class BaseDataset(Dataset):
    def __init__(self, species, vocab, subset, tokenize_method = "random", mask_prob = 0.0):        

        self.species_df = species

        # dict to map surveyId to speciesId

        self.species_vocab = vocab
        self.tokenize_method = tokenize_method
        self.mask_prob = mask_prob
        self.max_length = 100
        self.species_df = species.copy()
        self.subset = subset

        if self.species_vocab.binary_matrix == True :
            self.survey_to_species = {}
            for row in self.species_df.itertuples(index=False, name=None):
                survey_id = row[0]
                species_positions = np.where(row[1:])[0]
                species_indices = [
                    self.species_vocab.get_index(int(species_pos))
                    for species_pos in species_positions
                ]
                if len(species_indices) > 0:
                    self.survey_to_species[survey_id] = species_indices

        else :
            self.survey_to_species = (
                self.species_df.groupby("surveyId")["speciesId"]
                .unique()
                .apply(list)
                .to_dict()
            )

            self.survey_to_species = {
                survey_id: [self.species_vocab.get_index(s) for s in species_list if pd.notna(s)]
                for survey_id, species_list in self.survey_to_species.items()
                if len(species_list) > 0
            }

        self.surveys = [
            survey_id for survey_id in self.species_df.iloc[:, 0].unique().tolist()
            if survey_id in self.survey_to_species and len(self.survey_to_species[survey_id]) > 0
        ]
        self.species_df = self.species_df[self.species_df.iloc[:, 0].isin(self.surveys)].copy()

        # test if max_species exceeds max_length
        if self.tokenize_method is not None and len(self.survey_to_species) > 0:
            max_species = max(len(s) for s in self.survey_to_species.values())
            if max_species > self.max_length:
                print(f"Warning: max_species {max_species} exceeds max_length {self.max_length}. Consider increasing max_length.")

    def __len__(self):
        return len(self.surveys)
    
    def get_species(self, survey_id):

        # Map original species IDs to contiguous vocab indices safely
        species_tensor = torch.zeros(self.species_vocab.nb_species, dtype=torch.bool)

        if survey_id not in self.survey_to_species:
            print(f"Warning: survey_id {survey_id} not found, returning empty species tensor.")
            species_indices = []
        else: 
            species_indices = self.survey_to_species[survey_id]        
            species_tensor[species_indices] = 1


        if self.tokenize_method is not None:
            if self.tokenize_method  == "f_sort":
                species_indices = torch.tensor(species_indices)
                species_indices = species_indices[torch.argsort(torch.tensor([
                    self.species_vocab.species_counts.get(self.species_vocab.inv_vocab[int(s)], 0)
                    for s in species_indices.tolist()
                ], dtype=torch.float), descending=True)]
            elif self.tokenize_method == "random":
                species_indices = torch.tensor(species_indices)[torch.randperm(len(species_indices))]
            elif self.tokenize_method == "stocha":
                counts = [self.species_vocab.species_counts.get(self.species_vocab.inv_vocab[int(s)], 0) for s in species_indices]
                probs = torch.tensor(counts, dtype=torch.float)
                if probs.sum() == 0:
                    probs = torch.ones(len(species_indices), dtype=torch.float)
                probs /= probs.sum()
                if len(species_indices) != 0:
                    species_indices = torch.tensor(species_indices)[torch.multinomial(probs, num_samples=len(species_indices))]
            else :
                raise ValueError(f"Unknown tokenize method: {self.tokenize_method}")
        
            tokens = torch.full((self.max_length+1,), fill_value=self.species_vocab.pad_token, dtype=torch.long)
            species_indices = species_indices[:self.max_length]
            if self.mask_prob > 0.0:
                mask = torch.rand(len(species_indices)) < self.mask_prob
                species_indices[mask] = self.species_vocab.mask_token
            tokens[:len(species_indices)] = species_indices
            tokens[len(species_indices)] = self.species_vocab.eos_token
        
        if self.tokenize_method is not None:
            return (species_tensor, tokens)
        return (species_tensor,)


    def get_env(self, survey_id):
        return []

    def __getitem__(self, idx):
        survey_id = self.surveys[idx]
        spec = self.get_species(survey_id)                    
        env = self.get_env(survey_id)
        
        return survey_id, *spec, env



class TabularDataset(BaseDataset):
    def __init__(self, species, vocab, subset, x_dir = None, tokenize_method = "random", mask_prob = 0.0, index_column = "surveyId",var_cols = []):        
        super().__init__(species, vocab, subset, tokenize_method, mask_prob)

        self.emb_dir = x_dir
        emb_df = p.read_csv(x_dir)
        var_cols_id = [index_column]
        columns = emb_df.columns

        if type(var_cols) != list :
            var_cols = [var_cols]
        for name in var_cols :
            if type(name) == int :
                var_cols_id.append(columns[name])
            elif type(name) == str and name in columns :
                var_cols_id.append(name)
            else :
                raise Exception(f"Unknown tabular variable {name}. Valid names are : {columns}")
        self.emb_dir_df = emb_df[var_cols_id].set_index(index_column)

    def get_env(self, survey_id):
        assert survey_id in self.emb_dir_df.index, f"Missing ID : {survey_id}"
        env = [torch.tensor(self.emb_dir_df.loc[survey_id].to_numpy(dtype = np.float32))]
        return env
            