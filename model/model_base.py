import torch
import torch.nn as nn
from Hyperparameter import *
import torch.nn.functional as F


class TripletLoss2D(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss2D, self).__init__()
        self.margin = margin


    def forward(self, anchor, positive, negative):
        
        distance_pos = 1 - nn.functional.cosine_similarity(anchor, positive, dim=1)
        distance_neg = 1 - nn.functional.cosine_similarity(anchor, negative, dim=1)
        loss = torch.clamp(self.margin + distance_pos - distance_neg, min=0.0).mean()
        return loss,(distance_pos,distance_neg)

class CSDLoss2D(nn.Module):
    def __init__(self, margin=1.0,alpha=0.3,beta=0.2,gamma=0.5):
        super(CSDLoss2D, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def forward(self, anchor, positive, negative):


        # Triplet loss
        distance_pos = 1 - nn.functional.cosine_similarity(anchor, positive, dim=1)
        distance_neg = 1 - nn.functional.cosine_similarity(anchor, negative, dim=1)
        tri_loss = torch.clamp(self.margin + distance_pos - distance_neg, min=0.0).mean()

  
        cons_ap = torch.mean( torch.pow( distance_pos,2))
        cons_an = torch.mean( torch.pow( torch.clamp( self.margin - distance_neg,min=0.0 ),2))

        combined_loss = self.alpha * cons_ap + self.beta * cons_an + self.gamma * tri_loss
        
        return combined_loss,(distance_pos,distance_neg)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin


    def forward(self, anchor, positive, negative):
        # baseline 的 output 是 3 维数据,对-1 维求相似性,得到 [8,1024],再对 1024 求平均
        distance_pos = 1 - nn.functional.cosine_similarity(anchor, positive, dim=2).mean(dim=1)
        distance_neg = 1 - nn.functional.cosine_similarity(anchor, negative, dim=2).mean(dim=1)
        loss = torch.clamp(self.margin + distance_pos - distance_neg, min=0.0).mean()
        return loss,(distance_pos,distance_neg)
    
class CSDLoss(nn.Module):
    def __init__(self, margin=1.0,alpha=0.3,beta=0.2,gamma=0.5):
        super(CSDLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def forward(self, anchor, positive, negative):


        distance_pos = 1 - nn.functional.cosine_similarity(anchor, positive, dim=2).mean(dim=1)
        distance_neg = 1 - nn.functional.cosine_similarity(anchor, negative, dim=2).mean(dim=1)
        tri_loss = torch.clamp(self.margin + distance_pos - distance_neg, min=0.0).mean()


        cons_ap = torch.mean( torch.pow( distance_pos,2))
        cons_an = torch.mean( torch.pow( torch.clamp( self.margin - distance_neg,min=0.0 ),2))

        combined_loss = self.alpha * cons_ap + self.beta * cons_an + self.gamma * tri_loss
        
        return combined_loss,(distance_pos,distance_neg)
    



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Parameter(torch.rand(hidden_size))

    def forward(self, x): 
        u = torch.tanh(self.W(x))
        attention_weights = F.softmax(torch.matmul(u, self.context_vector), dim=1)
        attended_x = torch.sum(x * attention_weights.unsqueeze(2), dim=1)
        return attended_x, attention_weights
    
class Crosscode2vec(nn.Module):
    def __init__(self,p,t,e,
                 token_vocab_weight,node_vocab_weight,
                 k=256,h=128,path_atten_heads=2,ctxt_atten_heads=8
                 ):
        super(Crosscode2vec,self).__init__()

        self.token_vocab_weight = token_vocab_weight
        self.node_vocab_weight = node_vocab_weight

        self.pathfc = nn.Linear(2*h,k) 
        self.blstm = nn.LSTM(e,h, bidirectional=True, batch_first=True)  
        self.patht = nn.Tanh()

        self.tokenfc = nn.Linear(t*2*e,k) 
        self.tokent = nn.Tanh()

        self.pathatten2 = Attention(2*h)
        self.lstm = nn.LSTM(2*k,2*h, bidirectional=True, batch_first=True)  

        self.ctxtatten = Attention(4*h)
        self.outputfc = nn.Linear(h*4,e)


    def forward(self,astvector):

        b = astvector.shape[0]
        c = astvector.shape[1]
        token, path = astvector.split([token_length*2, path_length], dim=2) 

        token_emb = torch.tensor(self.token_vocab_weight[token]).to(device) 
        path_emb = torch.tensor(self.node_vocab_weight[path]).to(device)

        path_fc = path_emb.reshape(-1,path_emb.shape[-2],path_emb.shape[-1]) 
        

        path_fc,(phn,pcn) = self.blstm(path_fc)
        path_att,path_att_w = self.pathatten2(path_fc) 


        path_att = self.patht( self.pathfc(path_att.reshape(b,c,-1)))  
        token_fc = self.tokent( self.tokenfc(token_emb.reshape(b,c,-1)))  

        context = torch.concat((path_att,token_fc),dim=2)  

        context,_ = self.lstm(context) 
        ctxt,ctxt_w = self.ctxtatten(context) 
        # out-fc
        output = self.outputfc(ctxt) 

        return output
    
### 


class Code2Vec(nn.Module):
    def __init__(self,value_vocab_num,d,path_vocab_num,atten_heads):
        super(Code2Vec, self).__init__()
        self.value_emb = nn.Embedding(value_vocab_num,d)  
        self.path_emb = nn.Embedding(path_vocab_num,d) 
        self.fc = nn.Linear((token_length*2+path_length)*d, d)  
        self.atten = Attention(d)
      

    def forward(self,tpt):
   
        b = tpt.shape[0]
        c = tpt.shape[1]


        xs,xt,pj = tpt.split([token_length, token_length, path_length], dim=2) 
        xs_emb = self.value_emb(xs.to(device)) 
        pj_emb = self.path_emb(pj.to(device)) 
        xt_emb = self.value_emb(xt.to(device)) 
     
        ci = torch.concat((xs_emb,pj_emb,xt_emb),dim=2) 


        ci_ = torch.tanh(self.fc(ci.reshape(b,c,-1)))  
       
  
        att,att_w = self.atten(ci_) 

        return att  
    
class Code2seqEncoder(nn.Module):
    def __init__(self, path_vocab_num, token_vocab_num, p, e, h=64):
        super(Code2seqEncoder, self).__init__()
        self.pathembedding = nn.Embedding(path_vocab_num, e)
        self.tokenembedding = nn.Embedding(token_vocab_num, e)
        self.fc0 = nn.Linear(p * e, e)
        self.lstm = nn.LSTM(e, h, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * e, e)

    def forward(self, x):
        b = x.shape[0]
        c = x.shape[1]
        xs, pj = x.split([token_length * 2, path_length], dim=2)
        path_embedded = self.pathembedding(pj.to(device))
        ts_embedded = torch.sum(self.tokenembedding(xs.to(device)), dim=2)

        path_embedded = self.fc0(path_embedded.reshape(b, c, -1))
        lstm_outputs, (hidden_state, cell_state) = self.lstm(path_embedded)
        rep = torch.concat((lstm_outputs, ts_embedded), dim=2)

        encoder_outputs = self.fc(rep)

        return encoder_outputs

class HAN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(HAN, self).__init__()
        self.pathemb = nn.Embedding(vocab_size,embed_size) # path vocab_size
        self.word_gru = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)  # e,h
        self.word_attention = Attention(hidden_size * 2)
        
        self.sentence_gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.sentence_attention = Attention(hidden_size * 2)
        
        self.fc = nn.Linear(hidden_size * 2, embed_size)

    def forward(self, x):

        b = x.shape[0]
        c = x.shape[1]
        xs,pj = x.split([token_length*2, path_length], dim=2)  
        
        path_embedded = self.pathemb(pj.to(device))  

        word_output, _ = self.word_gru(path_embedded.reshape(-1,path_embedded.shape[-2],path_embedded.shape[-1]) ) 
        word_output, _ = self.word_attention(word_output)  

        sentence_output, _ = self.sentence_gru(word_output.reshape(b,c,-1)) 
        sentence_output, _ = self.sentence_attention(sentence_output)  

       
        output = self.fc(sentence_output) 

        return output
    



class Crosscode2vec_emb(nn.Module):
    def __init__(self,p,t,e,
                 token_vocab_weight,node_vocab_weight,
                 k=256,h=128,path_atten_heads=2,ctxt_atten_heads=8,node_vocab_num=173,token_vocab_num=3376
                 ):
        super(Crosscode2vec_emb,self).__init__()


        self.pathemb = nn.Embedding(node_vocab_num,e)
        self.tokeemb = nn.Embedding(token_vocab_num,e)

        self.pathfc = nn.Linear(2*h,k) 
        self.blstm = nn.LSTM(e,h, bidirectional=True, batch_first=True)  
        self.patht = nn.Tanh()

        self.tokenfc = nn.Linear(t*2*e,k)  
        self.tokent = nn.Tanh()

        self.pathatten2 = Attention(2*h)
        self.lstm = nn.LSTM(2*k,2*h, bidirectional=True, batch_first=True) 
    
        self.ctxtatten = Attention(4*h)
        self.outputfc = nn.Linear(h*4,e)


    def forward(self,astvector):
       
        b = astvector.shape[0]
        c = astvector.shape[1]
        token, path = astvector.split([4, 32], dim=2) 

        token_emb = self.tokeemb(token.to(device)) 
        path_emb = self.pathemb(path.to(device)) 

     
        path_fc = path_emb.reshape(-1,path_emb.shape[-2],path_emb.shape[-1]) 
        
    
        path_fc,(phn,pcn) = self.blstm(path_fc) 
        path_att,path_att_w = self.pathatten2(path_fc) 


        path_att = self.patht( self.pathfc(path_att.reshape(b,c,-1)))  
        token_fc = self.tokent( self.tokenfc(token_emb.reshape(b,c,-1)))  

        context = torch.concat((path_att,token_fc),dim=2)  

        context,_ = self.lstm(context) 
        ctxt,ctxt_w = self.ctxtatten(context) 
        output = self.outputfc(ctxt) 

        return output

