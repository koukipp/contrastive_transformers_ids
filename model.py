import torch
import torch.nn.functional as F

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
      super(EmbeddingLayer, self).__init__()
      self.position_embeddings = torch.nn.Embedding(34, hidden_dim)

      self.embedding_layer = torch.nn.ModuleList([torch.nn.Embedding(65539, 51) for i in range(3)])

      self.time_layer = torch.nn.Linear(1,51,bias=False)
      self.time_layer_2 = torch.nn.Linear(128,51)

      self.packet_layer = torch.nn.Linear(1,51,bias=False)
      self.packet_layer_2 = torch.nn.Linear(128,51)


      self.proj_layer = torch.nn.Linear(51 * 5, hidden_dim)
      torch.nn.init.xavier_uniform_(self.proj_layer.weight)
      self.proj_layer.bias.data.fill_(0)
      self.activ = torch.nn.ReLU()
      self.norm_3 = torch.nn.LayerNorm(256)
      self.norm_2 = torch.nn.LayerNorm(45 * 4)
      self.norm_1 = torch.nn.LayerNorm(273,eps=1e-12)
      self.dropout = torch.nn.Dropout(0.1)
    
    
    def forward(self, input):
        position = torch.arange(input.shape[1], dtype=torch.long, device="cpu")
        position = position.unsqueeze(0).expand((input.shape[0],input.shape[1]))

        list_emb = self.time_layer(input[:, :, [0]])
        packet_emb = self.packet_layer(input[:, :, [4]])
        list_emb_2 = [self.embedding_layer[i-1](input[:, :, i].int()) for i in range(1,input.shape[-1] - 1)]

        embed_tokens = self.proj_layer(torch.cat((list_emb,packet_emb,torch.cat(list_emb_2,dim=2)),dim=2))
        hidden_states = embed_tokens + self.position_embeddings(position)
        hidden_states = hidden_states
        return hidden_states
  

class OutputLayer(torch.nn.Module):
  def __init__(self, hidden_dim):
    super(OutputLayer, self).__init__()
    self.activ = torch.nn.ReLU()
    self.linear =  torch.nn.Linear(hidden_dim,hidden_dim)
    self.linear_2 =  torch.nn.Linear(hidden_dim,hidden_dim)
  
  def forward(self, input):
      cls = self.linear_2(self.activ(self.linear(input)))
      return cls

class CLSLayer(torch.nn.Module):
  def __init__(self, hidden_dim):
    super(CLSLayer, self).__init__()
    self.activ = torch.nn.ReLU()
    self.sig_activ = torch.nn.Sigmoid()
    self.linear =  torch.nn.Linear(hidden_dim,4*hidden_dim)
    self.linear_2 =  torch.nn.Linear(4*hidden_dim,hidden_dim)
    self.linear_3 =  torch.nn.Linear(hidden_dim,1)
  
  def forward(self, input):
      cls = self.sig_activ(self.linear_3(self.linear_2(self.activ(self.linear(input)))))
      return cls
  

class BERT(torch.nn.Module):
  def __init__(self, hidden_dim):
    super(BERT, self).__init__()
    self.norm = torch.nn.LayerNorm(768)
    self.embed = EmbeddingLayer(hidden_dim)
    self.encoder_layer = torch.nn.TransformerEncoderLayer(hidden_dim, 4, 256*4, batch_first=True, activation = 'gelu')
    self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=4)
    self.out_layer = OutputLayer(hidden_dim)
    self.cls_layer = CLSLayer(hidden_dim)
  
  def forward(self, input_1, input_2, mask):
      embed_1 = self.embed(input_1)
      enc = self.encoder(embed_1,src_key_padding_mask=mask)
      cls_1 = self.out_layer(enc[:, 0, :])
      
      embed_2 = self.embed(input_2) 
      enc = self.encoder(embed_2,src_key_padding_mask=mask)
      cls_2 = self.out_layer(enc[:, 0, :])
      return cls_1,cls_2
  
  def embeddings(self, input, mask):
      embed = self.embed(input)
      enc = self.encoder(embed,src_key_padding_mask=mask)
      return enc[:, 0, :]

  def embeddings_cls(self, input, mask):
      embed = self.embed(input)
      enc = self.encoder(embed,src_key_padding_mask=mask)
      return self.cls_layer(enc[:, 0, :])