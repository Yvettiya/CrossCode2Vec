
from database import MySQLiteDB
import os
import utils
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler,BatchSampler
from Hyperparameter import datarootpath,vector_dir,encoded_dir,content_lens
import json
import numpy as np


def loadAST(path):
        
    ast = utils.get_AST(path)
    vector = utils.vectorizeAST(ast)
    return vector

def word2vec_collate_fn(batch):
  
    contexts = []
    targets = []
    for context, target in batch:
        contexts.append(torch.tensor(context))
        targets.append(target)
    return torch.stack(contexts), torch.tensor(targets)


class AST():
    def __init__(self,info):
         
        self.type, self.name, self.proj = info
        self.path = os.path.join(datarootpath, self.type, self.proj, self.name)
        self.encodeast = utils.get_AST(self.path)
        self.vector = utils.vectorizeAST(self.encodeast)



class APCDataset(Dataset):
    def __init__(self,infolist):
        super().__init__()
        self.info = list(set(infolist))
        

    def __getitem__(self, index):

        info = self.info[index]
        ast = AST(info)
        return torch.tensor(ast.vector)
    
    def __len__(self):
        return len(self.info)
    
    def gettype(self,index):
        return self.info[index][0]
    
    def getfuncname(self,index):
        return self.info[index][1]
    
    def getprojname(self,index):
        return self.info[index][2]

    def find_by_info(self,astinfo):

        try:
            index = self.info.index(astinfo)
        except ValueError:
            index = -1
        return index
    
    def fuzzy_search(self,searchinfo):

        res = []
        for i in range(len(self.info)):
            if searchinfo in self.info[i]:
                res.append(i)
        return res

    def findmatching(self,index):
       
        target_type = 'src' if self.info[index][0] == 'bin' else 'bin'
        target_info = tuple([target_type]+list(self.info[index][1:]))
        return self.find_by_info(target_info)



def collate4pairs_fn(batch):
    a,p,n = zip(*batch)
    return torch.stack(list(a+p+n))


def npload_encoded_vector(index):
    vec = np.load(os.path.join(encoded_dir,str(index)+'.npy'))

    if content_lens <1024:
        vec = vec[:content_lens, : ]
    return vec

class APCDataset4baseline(Dataset):
    
    def __init__(self,data_pairs,loader=npload_encoded_vector):
        super(APCDataset4baseline).__init__()
        self.loader = loader
        self.data_pairs = data_pairs
        
    def __getitem__(self, index):
        data_pair = self.data_pairs[index]
        return torch.from_numpy(np.array([self.loader(data_id) for data_id in data_pair]))
    
    def getitem_id(self,index):
        data_pair = self.data_pairs[index]
        return data_pair
    
    def __len__(self):
        return len(self.data_pairs)
    
def npload_encoded_vector(index):
    vec = np.load(os.path.join(encoded_dir,str(index)+'.npy'))

    if content_lens <1024:
        vec = vec[:content_lens, : ]
    return vec

