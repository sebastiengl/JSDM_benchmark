import torch

class BaseSampler():
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def sample(self, y, nb = 1, survey_id = None):
        pass


class TopkSampler(BaseSampler):
    def __init__(self, model, vocab, k):
        super().__init__(model, vocab)
        self.k = k

    def sample(self, y, nb = 1, survey_id = None):
        self.model.eval()
        lgt = self.model(y, survey_id=survey_id)
        index = torch.topk(lgt, self.k).indices
        samples = torch.zeros_like(lgt)
        samples = samples.scatter_(1, index, 1).unsqueeze(1).expand(-1, nb, -1)
        self.model.train()
        return samples


class MarginalSampler(BaseSampler):
    def __init__(self, model, vocab):
        super().__init__(model, vocab)

    def sample(self, y, nb = 1, survey_id = None):
        self.model.eval()
        lgt = self.model(y, survey_id=survey_id)
        prob = lgt.sigmoid()
        # return nb bernouilling sampling from prob
        samples = torch.bernoulli(prob.unsqueeze(1).expand(-1,nb, -1))
        self.model.train()
        return samples

class MarginalStocSampler(BaseSampler):
    def __init__(self, model, vocab):
        super().__init__(model, vocab)

    def sample(self, y, nb = 1) :

        self.model.eval()

        B = y[0].shape[0]
        samples = torch.zeros((B, nb, self.vocab.nb_species), device=y[0].device)

        with torch.no_grad():
            for j in range(nb):
                lgt = self.model(y)
                prob = lgt.sigmoid()
                samples[:,j] = torch.bernoulli(prob)
        self.model.train()
        return samples


class SSESampler(BaseSampler):
    def __init__(self, model, vocab):
        super().__init__(model, vocab)

    def sample(self, y, nb = 1, survey_id = None):
        self.model.eval()
        lgt = self.model(y, survey_id=survey_id)
        rs = lgt.sigmoid().sum(dim=1).round().int()
        rs_max = rs.max().item()
        
        values, index = torch.topk(lgt, rs_max, dim=1)

        rank_range = torch.arange(rs_max, device=lgt.device).unsqueeze(0)  # (1, k_max)
        mask = (rank_range < rs.unsqueeze(1)).float()  # (B, k_max)

        samples = torch.zeros_like(lgt)
        samples = samples.scatter_(1, index, mask).unsqueeze(1).expand(-1, nb, -1)
        self.model.train()
        return samples




class SSEStocSampler(BaseSampler):
    def __init__(self, model, vocab):
        super().__init__(model, vocab)

    def sample(self, y, nb = 1):

        self.model.eval()
        # check if y is a list, TO DO  formalize and generalize
        if isinstance(y, list):
            B = y[0].shape[0]
            samples = torch.zeros((B, nb, self.vocab.nb_species), device=y[0].device)
        else :
            B = y.shape[0]
            samples = torch.zeros((B, nb, self.vocab.nb_species), device=y.device)

        with torch.no_grad():
            for j in range(nb):
                lgt = self.model(y)
                rs = lgt.sigmoid().sum(dim=1).round().int()
                rs_max = rs.max().item()
                
                values, index = torch.topk(lgt, rs_max, dim=1)

                rank_range = torch.arange(rs_max, device=lgt.device).unsqueeze(0)  # (1, k_max)
                mask = (rank_range < rs.unsqueeze(1)).float()  # (B, k_max)
                samples[:,j] = torch.zeros_like(lgt).scatter_(1, index, mask).float() 
        self.model.train()
        return samples


class AutoRegSampler(BaseSampler):
    def __init__(self, model, vocab, max_length, selection = "greedy", T = 1):
        super().__init__(model, vocab)
        self.selection = getattr(self, selection)
        self.max_length = max_length
        self.T = T #temperature for stocha, TO CHANGE

    def greedy(self, lgt, invalid_tokens):
        scores = lgt[-1].clone()
        if invalid_tokens:
            scores[list(invalid_tokens)] = float("-inf")
        pred_token = torch.argmax(scores).item()
        invalid_tokens.add(pred_token)
        return pred_token
    
    def stocha(self, lgt, invalid_tokens):
        scores = lgt[-1].clone()
        if invalid_tokens:
            scores[list(invalid_tokens)] = float("-inf")
        prob = torch.softmax(scores/self.T, dim=0)
        pred_token = torch.multinomial(prob, num_samples=1).item()
        invalid_tokens.add(pred_token)
        return pred_token

    def sample(self, y, nb = 1, survey_id = None):
        self.model.eval()
        B = y[0].shape[0]
        samples = torch.zeros((B, nb, self.vocab.nb_species), device=y[0].device)
        with torch.no_grad():
            for j in range(nb):
                for i in range(B):
                    pred = torch.full((self.max_length + 1,), fill_value=self.model.pad_token, dtype=torch.long, device= y[0].device)
                    invalid_tokens = set()
                    nb_spec = 0
                    lgt = self.model([y_n[i].unsqueeze(0) for y_n in y], pred[: nb_spec].unsqueeze(0))[-1]
                    pred_token = self.selection(lgt, invalid_tokens)
                    while pred_token != self.model.eos_token and nb_spec < self.max_length :
                        samples[i, j, pred_token] = 1
                        pred[nb_spec] = pred_token
                        nb_spec += 1
                        lgt = self.model([y_n[i].unsqueeze(0) for y_n in y], pred[: nb_spec].unsqueeze(0))[-1]
                        pred_token = self.selection(lgt, invalid_tokens)
        self.model.train()
        return samples

