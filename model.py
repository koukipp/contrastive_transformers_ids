import torch
import torch.nn.functional as F
import math
import torch.nn.init as nn_init

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, hidden_dim, seq_len, num_features, num_columns, dict_size=65536):
        super(EmbeddingLayer, self).__init__()
        self.num_features = num_features
        self.num_columns = num_columns
        self.seq_len = seq_len
        self.cat_mask = torch.ones(num_features, dtype=torch.bool)
        self.cat_mask[num_columns] = False
        self.num_mask = ~self.cat_mask
        self.norm = torch.nn.LayerNorm(hidden_dim)
        
        self.cls_token = torch.nn.Parameter(torch.empty((1, 1, hidden_dim)))
        nn_init.kaiming_uniform_(self.cls_token, a=math.sqrt(5))

        proj_dim = hidden_dim // num_features

        self.position_embeddings = torch.nn.Embedding(seq_len + 1, hidden_dim)
        self.cat_emb_layer = torch.nn.ModuleList([torch.nn.Embedding(dict_size, proj_dim) for _ in range(num_features - len(num_columns))])
        self.num_emb_layer = torch.nn.ModuleList([torch.nn.Linear(1, proj_dim, bias=False) for _ in range(len(num_columns))])

        self.proj_layer = torch.nn.Linear(proj_dim * num_features, hidden_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.proj_layer.weight)
    
        
        

    def forward(self, input):
        num_input = input[:, :, self.num_mask]
        num_emb = torch.cat([self.num_emb_layer[i](num_input[:, :, [i]]) for i in range(num_input.shape[-1])], dim=2)

        cat_input = input[:, :, self.cat_mask].int()
        cat_emb = torch.cat([self.cat_emb_layer[i](cat_input[:, :, i]) for i in range(cat_input.shape[-1])], dim=2)

        embed_tokens = self.proj_layer(torch.cat((num_emb, cat_emb), dim=2))

        batch_size = input.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        embed_tokens = torch.cat((cls_tokens, embed_tokens), dim=1)

        position = torch.arange(embed_tokens.shape[1], dtype=torch.long, device=embed_tokens.device)
        position = position.unsqueeze(0).expand((batch_size, embed_tokens.shape[1]))

        hidden_states = embed_tokens + self.position_embeddings(position)

        return self.norm(hidden_states)
  

class OutputLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(OutputLayer, self).__init__()
        self.activ = torch.nn.ReLU()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = torch.nn.Linear(hidden_dim, hidden_dim)
  
    def forward(self, input):
        cls = self.linear_2(self.activ(self.linear(input)))
        return cls


class CLSLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(CLSLayer, self).__init__()
        self.activ = torch.nn.ReLU()
        self.sig_activ = torch.nn.Sigmoid()
        self.linear = torch.nn.Linear(hidden_dim, 4*hidden_dim)
        self.linear_2 = torch.nn.Linear(4*hidden_dim, hidden_dim)
        self.linear_3 = torch.nn.Linear(hidden_dim, 1)
  
    def forward(self, input):
        cls = self.sig_activ(self.linear_3(self.linear_2(self.activ(self.linear(input)))))
        return cls
  

class BERT(torch.nn.Module):
    def __init__(self, num_features, num_columns, hidden_dim=256, seq_len=32):
        super(BERT, self).__init__()
        self.embed = EmbeddingLayer(hidden_dim, seq_len, num_features, num_columns)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*4, batch_first=True, activation='gelu')
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.out_layer = OutputLayer(hidden_dim)
        self.cls_layer = CLSLayer(hidden_dim)
            
    
    def embeddings(self, input, mask):  
        embed = self.embed(input)
        
        batch_size = input.shape[0]
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=input.device)
        transformer_mask = torch.cat((cls_mask, mask), dim=1)
        
        enc = self.encoder(embed, src_key_padding_mask=transformer_mask)
        return enc[:, 0, :]
    
    
    def forward(self, orig, aug, mask):
        orig_emb = self.out_layer(self.embeddings(orig, mask))
        aug_emb = self.out_layer(self.embeddings(aug, mask))
        return orig_emb, aug_emb
    
    def embeddings_cls(self, input, mask):
        return self.cls_layer(self.embeddings(input, mask))


class NTXent(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):

        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)
        ).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        

        return loss