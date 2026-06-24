import torch 
import torch.nn as nn
import math
#EMBEDDING LAYER 

#to convert the words into TOKENs with specid IDs
#THIS IDS OF THE WORD ARE THEN CONVERTED INTO VECTORS/EMBEDDINGS WITH SPECIFIC
#DIMENSION OR SIZE
#MOSTLY OF  DIMENSION 512  
#EACH TOKEN (WORD PIECES ) HAS AN ID

class InputEmbeddings(nn.Module):
    def __init__(self,d_model: int,vocab_size : int):
        #d_model is the dimension of the model of dtype = int
        #vocab_size is the size of the vocabulary we will use
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        #x is the object which will be converted into the input embeddings
        return self.embedding(x) * math.sqrt(self.d_model)

#POSITIONAL ENCODING
#THIS IS TO CONVEY THE INFORMATION ABOUT THE POSITION OF EACH WORD
#EACH WORD IN THE SENTENCE THIS IS DONE BY
# ADDING ANOTHER VECTOR TO THE INPUT EMBEDDINGS OF THE SAME DIM/SIZE

class PostionalEncoding(nn.Module):
    def __init__(self,d_model:int,dropout: float = 0.1,max_len: int = 5000) -> None:
        #seq_len = maximum no. of tokens(words) the model can process at one time
        #dropout = Dropout is a regularization technique used to prevent overfitting 
        #in neural networks by randomly "turning off" a fraction of neurons during training.
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        #create a matrix of shape(seq_len,d_model)
        pe = torch.zeros(max_len,d_model)
        #create a vector of shape(seq_len)
        position = torch.arange(0,max_len,dtype = torch.float32).reshape(-1,1) #(seq_len,-1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model) )
        # apply the sin to even 
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        #for broadcasting
        self.register_buffer('pe',pe)
        #this way the tensor will be save in the file along with the state of the model
    
    def forward(self,x):
        x = x + self.pe[:, :x.shape[1], :].detach()
        # x.shape give the seq_len of the sentence
        return self.dropout(x)

#LAYER NORMALIZATION:
#batch of 3 items:
# for each item in the batch we calculale the mean and the variance
# LayerNorm fixes this by normalizing the inputs to have a mean of 0 and a variance of 1.


class LayerNormalization(nn.Module):
    def __init__(self,eps : float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #multiplicative identity 
        self.bias = nn.Parameter(torch.zeros(1)) #additive identity

    def forward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        #this is to take the mean till the last dimension
        var = x.var(dim = -1,keepdim = True,unbiased = False)
        return self.alpha * (x-mean)/ torch.sqrt(var + self.eps) + self.bias
    
# we can also use the nn.LayerNorm
class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff: int,dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)
    
    def forward(self,x):
        #(Batch,seq_len,d_model) --> (batch,seq_len,d_ff) --> (batch,seq_len,d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    

#MULTI HEAD ATTENTION:
# the input embeddings after positional encoding will then be 
# split or duplicated into three i.e Q: query, K:key, V: values
# then multiply by Wq,Wk,Wv to get Q',K',W' 
# this new set of embedding will then be divided into 8 heads along their dimesnions
# so each head is of 512/8  or d_model/head dimensions
# self attention is applied in each head and the resulting then concetenated 
# this concetenated head is then multiplied by  a vector named Wo
# this gives the result of the multi headed attention

class Multiheadattention(nn.Module):
    def __init__(self,d_model: int,h: int,dropout: float) ->None:
        super().__init__()
        self.d_model = d_model
        self.h  = h
        assert d_model % h == 0, "d_model is divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model) 
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(query,key,value,mask,dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) #(batch,h,seq_len,seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value),attention_scores
    


    def forward(self,q,k,v,mask):
        #if we want some words to not interact with some other words 
        # we mask them 
        # used in decoder
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # (batch,seq_len,d_model) --> (batch,seq_len,h,d_k) --> (batch,h,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        x,self.attention_scores = Multiheadattention.attention(query,key,value,mask,self.dropout)
        # (batch,h,seq_len,d_k) -- > (batch,seq_len,h,d_k) --> (batch,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)
        return self.w_o(x)
#RESIDUAL CONNECTION:

class ResidualConnection(nn.Module):
    def __init__(self,dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    #sublayer(...): The normalized data is passed into the sublayer function. 
    # The sublayer parameter is dynamic—
    # it will be your Multi-Head Attention block in one instance, 
    # and your Feed-Forward Network block in the next.
    # this is a pre LN code implementation

#Encoder block:
class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block: Multiheadattention,feedforward : FeedForward,dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feedforward = feedforward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2) ])
    def forward(self,x,src_mask):
        x = self.residual_connection[0](x,lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x,self.feedforward)
        return x 
class Encoder(nn.Module):
    def __init__(self,layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
#DECODER BLOCK:
class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block: Multiheadattention,cross_attention_block :Multiheadattention,feedforward : FeedForward,dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward = feedforward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        #src_mask = mask of the encoder
        #tgt_mask = mask of the decoder
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connection[1](x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connection[2](x,self.feedforward)
        return x
    
#DECODER : COMBINATION OF ALL THE DECODER BLOCKS
class Decoder(nn.Module):
    def __init__(self,layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()


    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
class ProjectionLayer(nn.Module):
    def __init__(self,d_model: int,vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        #(BATCH,SEQ_LEN,D_MODEL) --> (BATCH,SEQ_LEN,vocab_size)
        return torch.log_softmax(self.proj(x),dim = -1)

#TRANSFORMER BLOCK:
class Transformer(nn.Module):
    def __init__(self,encoder: Encoder,decoder: Decoder,src_embed: InputEmbeddings,tgt_embed: InputEmbeddings,src_pos:PostionalEncoding,tgt_pos:PostionalEncoding,projectionlayer : ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projectionlayer = projectionlayer

    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    def project(self,x):
        return self.projectionlayer(x)


def build_transformer(
    src_vocab_size: int, 
    tgt_vocab_size: int, 
    src_seq_len: int = 5000, 
    tgt_seq_len: int = 5000, 
    d_model: int = 512, 
    N: int = 6, 
    h: int = 8, 
    dropout: float = 0.1, 
    d_ff: int = 2048
) -> Transformer:
    
    # 1. Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # 2. Create the positional encoding layers
    src_pos = PostionalEncoding(d_model, dropout, 5000)
    tgt_pos = PostionalEncoding(d_model,dropout, 5000)
    
    # 3. Create the encoder blocks stack
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = Multiheadattention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    # 4. Create the decoder blocks stack
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = Multiheadattention(d_model, h, dropout)
        decoder_cross_attention_block = Multiheadattention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block, 
            decoder_cross_attention_block, 
            feed_forward_block, 
            dropout
        )
        decoder_blocks.append(decoder_block)
        
    # 5. Group the stacks into the master Encoder and Decoder modules
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # 6. Create the final vocabulary projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # 7. Assemble the full system into the master Transformer class container
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # 8. Initialize weights using Xavier Uniform (crucial for deep training stability)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer


    




 
    


    
        







