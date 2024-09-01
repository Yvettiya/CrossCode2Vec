
import numpy as np

import torch


tokenvocabnum = 3376
nodevocabnum = 173


window_size = 5
context_size = (window_size-1)//2
word_batch_size = 128
word_emb_dim = 128      
word_learning_rate = 0.05
word_num_epochs = 80


node_matrix = np.load('embedding_matrix_dataset100_2.npy')
token_matrix = np.load('token_embedding_matrix_dataset100.npy')


path_length = 32         
content_lens = 512         
token_length = 2       


real_batchsize = 8            


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


margin = 1.8

alpha = 0.3
beta = 0.2
gamma = 0.5

learningrate = 0.001

num_epochs = 100



train_ratio = 0.7 
val_ratio = 0.2  
test_ratio = 0.1  


similarity_threshold = 0.5  
global_best_f1 = 0.4