import utils
from Hyperparameter import *

from model4ablation import *

from model_base import *

from trainer import *

import argparse

def build_and_train_model(epochnum, learningrate, margin, alpha, beta, datasetnum,model,loss,earlystopping,savepath,tasktype,ablation,phrase):

    print(f"Building and training model with parameters:")
    print(f"- Epochs: {epochnum}")
    print(f"- Learning Rate: {learningrate}")
    print(f"- margin: {margin}")
    print(f"- Alpha: {alpha}")
    print(f"- Beta: {beta}")
    print(f"- datasetnum: {datasetnum}")
    print(f"- model: {model}")
    print(f"- loss: {loss}")
    print(f"- earlystopping: {earlystopping}")
    print(f"- savepath: {savepath}")
    print(f"- tasktype: {tasktype}")
    print(f"- ablation: {ablation}")


if __name__ == '__main__':


    # 创建解析器
    parser = argparse.ArgumentParser(description='Example of a Python script with command-line arguments.')

    # 添加参数
    parser.add_argument('--epochnum', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--datasetnum', type=int, default=5000, help='Number of dataset(all of train,valid,test)')
    parser.add_argument('--learningrate', type=float, default=0.0015, help='Learning rate for training')
    parser.add_argument('--margin', type=float, default=1.1, help='margin hyperparameter')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha hyperparameter')
    parser.add_argument('--beta', type=float, default=0.2, help='Beta hyperparameter')
    parser.add_argument('--model', type=str, default='default_model', help='model type')
    parser.add_argument('--loss', type=str, default='default_loss', help='loss type')
    parser.add_argument('--earlystopping', type=bool, default=True, help='earlystopping True or False')
    parser.add_argument('--savepath', type=str, default='results', help='save path dir')
    parser.add_argument('--tasktype', type=str, default='cross', help='tasktype=[cross,src,bin]')
    parser.add_argument('--ablation', type=bool, default=False, help='whether do Ablation')

    args = parser.parse_args()



    if args.ablation == False:
    
        model_name = f'Model_{args.model}-{args.loss}_d-{args.datasetnum}_lr-{args.learningrate}_m-{args.margin}_a-{args.alpha}_b-{args.beta}_t-{args.tasktype}'


        if args.model == 'default_model':
            model = Crosscode2vec(path_length,token_length,word_emb_dim,utils.token_matrix,utils.node_matrix).to(device)
        elif args.model == 'code2vec':
            model = Code2Vec(3376,128,173,1).to(device)
        elif args.model == 'code2seq':
            model = Code2seqEncoder(173,3376,path_length,word_emb_dim).to(device)
        elif args.model == 'han':
            model = HAN(173,128,32).to(device)
        else:
            print('error model type!')

        if args.loss == 'default_loss':
            loss = TripletLoss(margin=args.margin)
        elif args.loss == 'csd':
            loss = CSDLoss(margin=args.margin,alpha=args.alpha,beta=args.beta,gamma=1-args.alpha-args.beta)
        elif args.loss == 'tripletloss2D':
            loss = TripletLoss2D(margin=args.margin)        
        elif args.loss == 'CSD2D':
            loss = CSDLoss2D(margin=args.margin,alpha=args.alpha,beta=args.beta,gamma=1-args.alpha-args.beta)
        else:
            print('error loss type!')

        if args.tasktype == 'cross':
            train_loader,valid_loader,test_loader = get_dataloader(args.datasetnum)
        elif args.tasktype == 'bin':
            train_loader,valid_loader,test_loader = get_dataloader_binary(args.datasetnum)
        elif args.tasktype == 'src':
            train_loader,valid_loader,test_loader = get_dataloader_srouce(args.datasetnum)
        else:
            print('error task type!')

  
    else: 
        model_name = f'ABLATION_{args.model}-{args.loss}_d-{args.datasetnum}_lr-{args.learningrate}_m-{args.margin}_a-{args.alpha}_b-{args.beta}_t-{args.tasktype}'

        if args.model == 'origin':
            model = Crosscode2vec(path_length,token_length,word_emb_dim,utils.token_matrix,utils.node_matrix).to(device)
        elif args.model == 'lackemb':
            model = Crosscode2vec_lackemb(path_length,token_length,word_emb_dim,utils.token_matrix,utils.node_matrix,tokenvocabnum,nodevocabnum).to(device)
        elif args.model == 'lackPA':
            model = Crosscode2vec_lackPA(path_length,token_length,word_emb_dim,utils.token_matrix,utils.node_matrix).to(device)
        elif args.model == 'lackCA':
            model = Crosscode2vec_lackCA(path_length,token_length,word_emb_dim,content_lens,utils.token_matrix,utils.node_matrix).to(device)
        elif args.model == 'LackAtten':
            model = Crosscode2vec_LackAtten(path_length,token_length,word_emb_dim,content_lens,utils.token_matrix,utils.node_matrix).to(device)
        
        else:
            print('error model type!')

        if args.loss == 'CSD2D':
            loss = CSDLoss2D(margin=args.margin,alpha=args.alpha,beta=args.beta,gamma=1-args.alpha-args.beta)
        elif args.loss == 'tripletloss2D':
            loss = TripletLoss2D(margin=args.margin) 
        else:
            print('error loss type!')
        
        
        if args.tasktype == 'src':
            train_loader,valid_loader,test_loader = get_dataloader_srouce(args.datasetnum)
        else:
            train_loader,valid_loader,test_loader = get_dataloader(args.datasetnum)



    optimizer = torch.optim.Adam(model.parameters(),lr=args.learningrate)

    if args.earlystopping:
        run_model_with_earlystopping(model,loss,optimizer,train_loader,valid_loader,test_loader,model_name,args.savepath,args.epochnum)
    else:
        run_model(model,loss,optimizer,train_loader,valid_loader,test_loader,model_name,args.savepath,args.epochnum)




