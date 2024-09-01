import torch
import torch.nn as nn
from mylogger import logger
from Hyperparameter import *

from model_base import Attention

class Crosscode2vec_lackemb(nn.Module):
    def __init__(self, p, t, e, token_vocab_weight, node_vocab_weight, token_vocab_num, node_vocab_num, k=256, h=128, path_atten_heads=2, ctxt_atten_heads=8):
        super(Crosscode2vec_lackemb, self).__init__()
        self.pathemb = nn.Embedding(node_vocab_num, e)
        self.tokeemb = nn.Embedding(token_vocab_num, e)

        self.pathfc = nn.Linear(2 * h, k)
        self.blstm = nn.LSTM(e, h, bidirectional=True, batch_first=True)
        self.patht = nn.Tanh()

        self.tokenfc = nn.Linear(t * 2 * e, k)
        self.tokent = nn.Tanh()

        self.pathatten2 = Attention(2 * h)
        self.lstm = nn.LSTM(2 * k, 2 * h, bidirectional=True, batch_first=True)
        self.ctxtatten = Attention(4 * h)
        self.outputfc = nn.Linear(h * 4, e)

    def forward(self, astvector):
        b = astvector.shape[0]
        c = astvector.shape[1]
        token, path = astvector.split([4, 32], dim=2)
        token_emb = self.tokeemb(token.to(device))
        path_emb = self.pathemb(path.to(device))

        path_fc = path_emb.reshape(-1, path_emb.shape[-2], path_emb.shape[-1])
        path_fc, (phn, pcn) = self.blstm(path_fc)
        path_att, path_att_w = self.pathatten2(path_fc)

        path_att = self.patht(self.pathfc(path_att.reshape(b, c, -1)))
        token_fc = self.tokent(self.tokenfc(token_emb.reshape(b, c, -1)))

        context = torch.concat((path_att, token_fc), dim=2)
        context, _ = self.lstm(context)
        ctxt, ctxt_w = self.ctxtatten(context)

        output = self.outputfc(ctxt)

        return output

class Crosscode2vec_lackPA(nn.Module):
    def __init__(self, p, t, e, token_vocab_weight, node_vocab_weight, k=256, h=128, path_atten_heads=2, ctxt_atten_heads=8):
        super(Crosscode2vec_lackPA, self).__init__()
        self.token_vocab_weight = token_vocab_weight
        self.node_vocab_weight = node_vocab_weight

        self.blstm = nn.LSTM(e, h, bidirectional=True, batch_first=True)
        self.tokenfc = nn.Linear(t * 2 * e, k)
        self.tokent = nn.Tanh()
        self.pathlinear_layer = nn.Linear(p * h * 2, k)

        self.lstm = nn.LSTM(2 * k, 2 * h, bidirectional=True, batch_first=True)
        self.ctxtatten = Attention(4 * h)
        self.outputfc = nn.Linear(h * 4, e)

    def forward(self, astvector):
        b = astvector.shape[0]
        c = astvector.shape[1]
        token, path = astvector.split([4, 32], dim=2)
        token_emb = torch.tensor(self.token_vocab_weight[token]).to(device)
        path_emb = torch.tensor(self.node_vocab_weight[path]).to(device)

        path_fc = path_emb.reshape(-1, path_emb.shape[-2], path_emb.shape[-1])
        path_fc, (phn, pcn) = self.blstm(path_fc)

        path_fc_reshaped = path_fc.reshape(-1, path_fc.size(1) * path_fc.size(2))
        path_fc_res = self.pathlinear_layer(path_fc_reshaped)
        path_fc_res = path_fc_res.reshape(b, c, -1)

        token_fc = self.tokent(self.tokenfc(token_emb.reshape(b, c, -1)))

        context = torch.concat((path_fc_res, token_fc), dim=2)

        context, _ = self.lstm(context)
        ctxt, ctxt_w = self.ctxtatten(context)

        output = self.outputfc(ctxt)

        return output
    

class Crosscode2vec_lackCA(nn.Module):
    def __init__(self, p, t, e, c, token_vocab_weight, node_vocab_weight, k=256, h=128, path_atten_heads=2, ctxt_atten_heads=8):
        super(Crosscode2vec_lackCA, self).__init__()
        self.token_vocab_weight = token_vocab_weight
        self.node_vocab_weight = node_vocab_weight

        self.pathfc = nn.Linear(2 * h, k)
        self.blstm = nn.LSTM(e, h, bidirectional=True, batch_first=True)
        self.patht = nn.Tanh()

        self.tokenfc = nn.Linear(t * 2 * e, k)
        self.tokent = nn.Tanh()

        self.pathatten2 = Attention(2 * h)
        self.lstm = nn.LSTM(2 * k, 2 * h, bidirectional=True, batch_first=True)
        self.outputfc = nn.Linear(h * 4 * c, e)

    def forward(self, astvector):
        b = astvector.shape[0]
        c = astvector.shape[1]
        token, path = astvector.split([4, 32], dim=2)
        token_emb = torch.tensor(self.token_vocab_weight[token]).to(device)
        path_emb = torch.tensor(self.node_vocab_weight[path]).to(device)

        path_fc = path_emb.reshape(-1, path_emb.shape[-2], path_emb.shape[-1])
        path_fc, (phn, pcn) = self.blstm(path_fc)
        path_att, path_att_w = self.pathatten2(path_fc)

        path_att = self.patht(self.pathfc(path_att.reshape(b, c, -1)))
        token_fc = self.tokent(self.tokenfc(token_emb.reshape(b, c, -1)))

        context = torch.concat((path_att, token_fc), dim=2)

        context, _ = self.lstm(context)

        context = context.reshape(b, -1)
        output = self.outputfc(context)

        return output    

class Crosscode2vec_LackAtten(nn.Module):
    def __init__(self, p, t, e, c, token_vocab_weight, node_vocab_weight, k=256, h=128):
        super(Crosscode2vec_LackAtten, self).__init__()
        self.token_vocab_weight = token_vocab_weight
        self.node_vocab_weight = node_vocab_weight

        self.blstm = nn.LSTM(e, h, bidirectional=True, batch_first=True)
        self.tokenfc = nn.Linear(t * 2 * e, k)
        self.tokent = nn.Tanh()
        self.pathlinear_layer = nn.Linear(p * h * 2, k)

        self.lstm = nn.LSTM(2 * k, 2 * h, bidirectional=True, batch_first=True)
        self.outputfc = nn.Linear(h * 4 * c, e)

    def forward(self, astvector):
        b = astvector.shape[0]
        c = astvector.shape[1]
        token, path = astvector.split([4, 32], dim=2)
        token_emb = torch.tensor(self.token_vocab_weight[token]).to(device)
        path_emb = torch.tensor(self.node_vocab_weight[path]).to(device)

        path_fc = path_emb.reshape(-1, path_emb.shape[-2], path_emb.shape[-1])
        path_fc, (phn, pcn) = self.blstm(path_fc)

        path_fc_reshaped = path_fc.reshape(-1, path_fc.size(1) * path_fc.size(2))
        path_fc_res = self.pathlinear_layer(path_fc_reshaped)
        path_fc_res = path_fc_res.reshape(b, c, -1)

        token_fc = self.tokent(self.tokenfc(token_emb.reshape(b, c, -1)))

        context = torch.concat((path_fc_res, token_fc), dim=2)

        context, _ = self.lstm(context)
        context = context.reshape(b, -1)
        output = self.outputfc(context)

        return output


