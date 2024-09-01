import sqlite3
import os
from database import MySQLiteDB
import pandas as pd
import re
import json
import numpy as np
import random

from Hyperparameter import datarootpath

def travel_code2vec(rootpath):

    conn = sqlite3.connect('./database.db')
    cursor = conn.cursor()
    
    
    for binsrc in sorted(list(os.listdir(rootpath))):
        
        for projinfo in sorted(list(os.listdir(os.path.join(rootpath,binsrc))) ):
            
            for funcname in sorted(list(os.listdir(os.path.join(rootpath,binsrc,projinfo)))):

                path = os.path.join(rootpath,binsrc,projinfo,funcname)

                data = (funcname, binsrc, projinfo, path)
                cursor.execute('INSERT INTO code2vec (funcname, type, labels, path) VALUES(?,?,?,?)', data)
                conn.commit()


    # 关闭连接
    conn.close()


def read_csv(filepath):

    try:
    
        pd_reader = pd.read_csv(filepath, na_filter=False)
    
        return [xi[1] for xi in pd_reader.values]
    
    except:
        
        return []
def read_csv_2(filepath):

    try:
    
        pd_reader = pd.read_csv(filepath, na_filter=False)
    
        return [xi for xi in pd_reader.values]
    
    except:
        
        return []
    
def read_csv_3(filepath):

    try:
    
        pd_reader = pd.read_csv(filepath, na_filter=False)
    
        return dict(pd_reader.values)
    
    except:
        
        return {}
    
def get_nodetype(funcpath):
    
     nodefilepath = os.path.join(funcpath,'c','node_types.csv')
     nodeinfo = read_csv_2(nodefilepath)
     return nodeinfo

def get_path(funcpath):
     
     pathfilepath = os.path.join(funcpath,'c','paths.csv')
     pathinfo = read_csv_2(pathfilepath)
     return pathinfo

def get_tokens(funcpath):
   
     nodefilepath = os.path.join(funcpath,'c','tokens.csv')
     nodeinfo = read_csv(nodefilepath)
     return nodeinfo

def get_nodetype_dict(funcpath):
     nodefilepath = os.path.join(funcpath,'c','node_types.csv')
     nodeinfo = read_csv_3(nodefilepath)
     return nodeinfo

def get_path_dict(funcpath):
     pathfilepath = os.path.join(funcpath,'c','paths.csv')
     pathinfo = read_csv_3(pathfilepath)
     return pathinfo

def get_tokens_dict(funcpath):
     nodefilepath = os.path.join(funcpath,'c','tokens.csv')
     nodeinfo = read_csv_3(nodefilepath)
     return nodeinfo

def get_nodevocab():
  
    with open('../dataset/re_embedding/node_embedding/node_vocabulary.json','r') as f:
        nodevocab = json.load(f)
    return nodevocab

def get_tokenvocab():
    
    with open('../dataset/re_embedding/token_embedding/token_vocabulary.json','r') as f:
        tokenvocab = json.load(f)
    return tokenvocab


# 加载词表
tokenvocab = get_tokenvocab()
nodevocab = get_nodevocab()

def rep_value_onehot(word):

    symbol_pattern = r'[^\w\s]'

    letter_pattern = r'\b[a-zA-Z]\b'

    variable_pattern = r'(int|bool|float|char|double|byte|long|unsigned|qword|void|dword|byte)\b'

    register_pattern = r'\b([er]ax|[er]bx|[er]cx|[er]dx|[er]si|[er]di)\b'

    
    
    if word in tokenvocab.keys():
        return tokenvocab[word]
    else:
        if re.match(symbol_pattern, word):
            return tokenvocab['<symbol>']
        elif re.match(letter_pattern, word):
            return tokenvocab['<singleletter>']
        elif re.match(variable_pattern, word):
            return tokenvocab['<variabletype>']
        elif re.match(register_pattern, word):
            return tokenvocab['<register>']
        else:
            return tokenvocab['<unk>']
        
def rep_node_onehot(word):
    if word in nodevocab.keys():
        return nodevocab[word]
    else:
        return nodevocab['UNK']



def get_pathcontents(funcpath):

    pathcontextfilepath = os.path.join(funcpath,'c','data','path_contexts.c2s')
    pathcontext = []
    try:
        with open(pathcontextfilepath,'r') as f:
            for eachline in f.readlines():
                pathcontext += eachline.strip().split(' ')[1:]
    except:
        pass
    return list(set(pathcontext))


def get_AST(funcpath):

    nodes = get_nodetype_dict(funcpath)
    paths = get_path_dict(funcpath)
    tokens = get_tokens_dict(funcpath)
    pathcontext = get_pathcontents(funcpath)

    if nodes == {} or paths == {} or tokens == {}:
        return []

    try:
        contents = []

        for pathcontextitem in pathcontext:
            cons = pathcontextitem.split(',')
            ts = tuple([rep_value_onehot(v) for v in tokens[int(cons[0])].split('|')])
            te = tuple([rep_value_onehot(v) for v in tokens[int(cons[2])].split('|')])
            ph = tuple([rep_node_onehot(nodes[int(n)]) for n in paths[int(cons[1])].split(' ')])


            contents.append((ts,ph,te))
    except:
        return []
    
    return list(set(contents))

def get_funcpaths(funcpath):

    nodes = get_nodetype_dict(funcpath)
    paths = get_path_dict(funcpath)
    pathcontext = get_pathcontents(funcpath)
    pathresults = []
    if paths == {}:
        return []
    for pathcontextitem in pathcontext:
        cons = pathcontextitem.split(',')
        ph = tuple([rep_node_onehot(nodes[int(n)]) for n in paths[int(cons[1])].split(' ')])

        pathresults.append((ph))

    return list(set(pathresults))

def get_functokens(funcpath):
    tokens = get_tokens_dict(funcpath)
    pathcontext = get_pathcontents(funcpath)

    if tokens == {}:
        return []

    contents = []

    for pathcontextitem in pathcontext:
        cons = pathcontextitem.split(',')
        ts = tuple([rep_value_onehot(v) for v in tokens[int(cons[0])].split('|')])
        te = tuple([rep_value_onehot(v) for v in tokens[int(cons[2])].split('|')])

        contents.append(ts)
        contents.append(te)

    return list(set(contents))

def get_matching_pairs():

    from collections import Counter
    database = MySQLiteDB('database.db')

    database.connect()
    all_funcs = database.select_columns_all('code2vec','funcname,labels')
    count = Counter(all_funcs)
    matching_pairs = []
    for item in count.keys():
        if count[item] >1:
            matching_pairs.append(item)
    database.close()
    return matching_pairs

def get_context(paths_collection,context_size):

    data_context = set()
    for nodeseq in paths_collection:

        for i in range(context_size, len(nodeseq) - context_size):
            context = tuple([nodeseq[j] for j in range(i - context_size, i + context_size + 1) if j != i])
            target = nodeseq[i] 
            data_context.add((context, target))

    return list(data_context)


def padorsplitpath(path,paddinglength,padding):
    if len(path)<paddinglength:
        path +=  [padding] * (paddinglength- len(path))
    else:
        path = path[:paddinglength]

    return path

from Hyperparameter import node_matrix,token_matrix,path_length,content_lens,token_length


nodepadding = nodevocab['PAD']

contentpadding = np.concatenate(
        (token_matrix[tokenvocab['<PAD>']].reshape(1, -1),
         token_matrix[tokenvocab['<PAD>']].reshape(1, -1),
         node_matrix[padorsplitpath([nodepadding],path_length,nodepadding)])
        , axis=0)

def vectorizeAST(ast):
    contents = []
    for ts,ph,te in ast:
        ts_vector = np.mean(token_matrix[list(ts)], axis=0)
        te_vector = np.mean(token_matrix[list(te)], axis=0)

        path_vector = node_matrix[padorsplitpath(list(ph),path_length,nodepadding)]
        content_matrix = np.concatenate((ts_vector.reshape(1, -1), te_vector.reshape(1, -1), path_vector), axis=0)

        contents.append(content_matrix)

    res = padorsplitpath(contents,content_lens,contentpadding)

    return np.array(res)

contentpadding_list = padorsplitpath([tokenvocab['<PAD>']],token_length,tokenvocab['<PAD>']) + \
            padorsplitpath([tokenvocab['<PAD>']],token_length,tokenvocab['<PAD>']) + \
            padorsplitpath([nodepadding],path_length,nodepadding)

def paddingAST(ast):
   
    contents = []
    for ts,ph,te in ast:
        ts_matrix = padorsplitpath(ts,token_length,tokenvocab['<PAD>'])
        te_matrix = padorsplitpath(te,token_length,tokenvocab['<PAD>'])
        path_matrix = padorsplitpath(list(ph),path_length,nodepadding)
        
        content_matrix = ts_matrix+te_matrix+path_matrix

        contents.append(content_matrix)

    res = padorsplitpath(contents,content_lens,contentpadding_list)

    return np.array(res)

def vectorizeAST2(ast):

    contents = []
    for ts,ph,te in ast:
        ts_vector = np.mean(token_matrix[list(ts)], axis=0)
        te_vector = np.mean(token_matrix[list(te)], axis=0)

        path_vector = node_matrix[padorsplitpath(list(ph),path_length,nodepadding)]
        content_matrix = np.concatenate((ts_vector.reshape(1, -1), te_vector.reshape(1, -1), path_vector), axis=0)

        contents.append(content_matrix)

    res = padorsplitpath(contents,content_lens,contentpadding)

    return res

def gen_datasetinfolist(pairs):

    res = []

    apn = []
    index = 0
    for item in pairs:
        res.append(tuple(['bin']+list(item)))
        res.append(tuple(['src']+list(item)))
        # positive.append((index))
        # negative.append((index,index+2))
        if index < len(pairs)*2 -2:
            apn.append((index,index+1,index+2))
        else:
            apn.append((index,index+1,0))

        index +=2
    return res,apn

def get_ast_encoded(funcpath):
    encoded_path = os.path.join(funcpath,'ast_encoded.json')
    if os.path.exists(encoded_path):
        with open(encoded_path,'r') as fp:
            return json.load(fp)
    else:
        return []


def gen_dataset_vectors(pairs):
  
    res = []
    index = 0
    apn = []
    ast_vectors = []
    for item in pairs:
        funcname = item[0]
        projname = item[1]
        src_path = os.path.join(datarootpath,'src',projname,funcname)
        bin_path = os.path.join(datarootpath,'bin',projname,funcname)

        src_encode = get_ast_encoded(src_path)
        bin_encode = get_ast_encoded(bin_path)

        if src_encode == [] or bin_encode == []:
            continue
        else:
            src_vector = vectorizeAST(src_encode)
            bin_vector = vectorizeAST(bin_encode)
            res.append(tuple(['bin']+list(item))) 
            res.append(tuple(['src']+list(item)))
            ast_vectors.append(bin_vector)
            ast_vectors.append(src_vector)
            apn.append((index,index+1,index+2))
            index +=2
    return res,apn,ast_vectors

def gen_dataset_info(pairs):
    res = []
    index = 0
    apn = []

    for item in pairs:
        funcname = item[0]
        projname = item[1]
        src_path = os.path.join(datarootpath,'src',projname,funcname)
        bin_path = os.path.join(datarootpath,'bin',projname,funcname)

        src_encode = get_ast_encoded(src_path)
        bin_encode = get_ast_encoded(bin_path)

        if src_encode == [] or bin_encode == []:
            continue
        else:
       
            res.append(tuple(['bin']+list(item)))  
            res.append(tuple(['src']+list(item)))

            apn.append((index,index+1,index+2))
            index +=2
    return res,apn

def locate_path(fname,pname,type):
    return os.path.join(datarootpath,type,pname,fname)

def gen_apn_from_ap(ap):

    apn_list = []
    for index in range(len(ap)):
        a,p = ap[index]
        if index<len(ap)-1:
            n = ap[index+1][0]
        else:
            n = ap[0][0]
        apn_list.append((a,p,n))
    return apn_list

def generate_triplets_batchall(data):
    triplets = []
    for element_id in range(len(data)):
        
        element = data[element_id]

        for i in range(len(element)-1):
            for j in range(i+1, len(element)):
                a = element[i]
                p = element[j]
                n = random.choice(data[random.choice([x for x in range(0, len(data)) if x != element_id])])
                triplets.append((a,p,n))
    return triplets


def id_path_map():
    database = MySQLiteDB('database.db')
    database.connect()
    return dict(database.select_columns_all('code2vec','id,path'))



def generate_triplets_batchall_src_bin(data):
    triplets = []
    for element_id in range(len(data)):
        a = random.choice( data[element_id][0]) 
        p_element = data[element_id][1]  

        for i in range(len(p_element)):
            p = p_element[i] 
          
            n = random.choice(random.choice(data[random.choice([x for x in range(0, len(data)) if x != element_id])]))

            triplets.append((a,p,n))
    return triplets



def generate_triplets_batchall_bin_src(data):
    triplets = []
    for element_id in range(len(data)):
        a = random.choice( data[element_id][0])  
        p_element = data[element_id][1] 

        for i in range(len(p_element)):
            p = p_element[i] 
          

            n = random.choice(random.choice(data[random.choice([x for x in range(0, len(data)) if x != element_id])]))

            triplets.append((a,p,n))
    return triplets
