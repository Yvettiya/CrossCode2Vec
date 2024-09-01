import torch
import torch.nn as nn
from mylogger import logger
from Hyperparameter import *
import numpy as np
from sklearn.metrics import precision_recall_curve, auc ,average_precision_score, classification_report,fbeta_score,accuracy_score,precision_score,recall_score
# import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os,json




def train(model, optimizer, dataloader, triplet_loss):
    model.train()
    correct = 0
    total = 0
    total_loss = 0.0

    losses = []

    for i, batch in enumerate(dataloader):
        model_out = model(batch)

        a_out, p_out, n_out = torch.split(model_out, [real_batchsize, real_batchsize, real_batchsize], dim=0)
        loss, (distance_pos, distance_neg) = triplet_loss(a_out, p_out, n_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for aa in torch.lt(distance_pos, distance_neg):
            if aa:
                correct += 1
            total += 1

        total_loss += loss.item()

        losses.append(loss.item())

    return total_loss / len(dataloader), losses, correct / total


def evalueated(model,dataloader,triplet_loss):
    '''
    output: 
        result_report: string
        pr_ccurve_for plt: tuple
    '''
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    all_pos = []
    all_neg = []
    gt_labels = []
    losses = []
    all_probs = []

    with torch.no_grad():
        for i,batch in enumerate(dataloader) :

            model_out = model(batch)

            a_out,p_out,n_out = torch.split(model_out,[real_batchsize,real_batchsize,real_batchsize],dim=0)

            exchange_ap = False
            import random
            import time
            random.seed(int(time.time()))
            if (random.randint(1,1000)+i) %2 == 0:
                p_out, n_out = n_out, p_out
                gt_labels.extend([1] * real_batchsize)
                exchange_ap = True

            else:
                gt_labels.extend([0] * real_batchsize)
            

            loss,(distance_pos,distance_neg) = triplet_loss(a_out,p_out,n_out)

            probs = nn.functional.softmax(torch.stack([2-distance_pos,2-distance_neg],dim=1),dim=1) 

            all_pos.extend(distance_pos.cpu().detach().numpy())
            all_neg.extend(distance_neg.cpu().detach().numpy())
            all_probs.extend(probs[:, 0].cpu().detach().numpy())

            for i in torch.lt(distance_pos, distance_neg):
                if i :
                    if not exchange_ap:
                        correct += 1
                else:
                    if exchange_ap:
                        correct +=1
                total += 1
            total_loss += loss.item()

            losses.append(loss.item())
        


        preds = [1 if pos > neg else 0 for pos, neg in zip(all_pos, all_neg)] 


        pcs_scoer = precision_score(gt_labels,preds,pos_label=0)
        rcl_score = recall_score(gt_labels,preds,pos_label=0)
        classify = classification_report(gt_labels,preds) 

        AP_score = average_precision_score(gt_labels,all_probs,pos_label=0)

        precision,recall,thresholds = precision_recall_curve(gt_labels,all_probs,pos_label=0)

        pr_auc = auc(recall,precision)

        f1_scores = f1_score(gt_labels,preds,pos_label=0)
        f2_scores = fbeta_score(gt_labels,preds,beta=2,pos_label=0)
        f05_scores = fbeta_score(gt_labels,preds,beta=0.5,pos_label=0)


        performance = {
            "Accuracy" : correct/total,
            "Precision" : pcs_scoer,
            "Recall" : rcl_score,
            "AverageLoss" : total_loss / len(dataloader),
            "AP_score" : AP_score,
            "PR_AUC": pr_auc,  
            "Classification": classify,      
            "matrix":{
                "PrecisionMatrix" : precision.tolist(),
                "RecalMatrix" : recall.tolist(),
                "ThresholdMatrix" : thresholds.tolist(),
                "F1scoreMatrix" : f1_scores.tolist(),
                "F2scoreMatrix": f2_scores.tolist(),
                "F0.5scoreMatrix": f05_scores.tolist(),
            },
            "losses" : losses
        }

        logger.info(gen_output_report(performance))

    return performance, np.asanyarray([precision, recall , pr_auc],dtype=object)


def gen_output_report(performanceinfo):

        output0 = f'''Accurancy = {performanceinfo['Accuracy']:.4f},Precision = {performanceinfo['Precision']:.4f},Recall = {performanceinfo['Recall']:.4f},Average loss = {performanceinfo['AverageLoss'] :.4f}\n'''

        
        output2 = f'''PR AUC score = {performanceinfo['PR_AUC']:.4f},average precision = {performanceinfo['AP_score']:.4f},Classification results: \n{performanceinfo['Classification']}''' 

        return output0+output2
  
def gen_output_report_simple(performanceinfo):
    
        output0 = f'''Accurancy = {performanceinfo['Accuracy']:.4f},Precision = {performanceinfo['Precision']:.4f},Recall = {performanceinfo['Recall']:.4f},Average loss = {performanceinfo['AverageLoss'] :.4f}\n'''


        return output0

def run_model(model,triplet_loss,optimizer,train_loader,valid_loader,test_loader,model_name,savepath='.',num_epochs=num_epochs):
    
    logger.info('********______________________________________********')

    mkresultdir(savepath)

    v_prcurves = []
    t_prcurves = []

    logger.info(f'START Training and Testing...\n MODEL {model_name} info :{model}')
    print(f'START Training and Testing...\n MODEL {model_name} info :{model}')

    performances = {"train":[],"valid":[],"test":[]}

    for epoch in range(num_epochs):
        train_loss,train_losses,train_acc = train(model, optimizer, train_loader, triplet_loss)
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}] Train Phrase: loss = {train_loss:.4f} ,acc = {train_acc:.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Phrase: loss = {train_loss:.4f} ,acc = {train_acc:.4f}')
        
        performances['train'].append({'AverageLoss':train_loss,'losses':train_losses,'Accuracy':train_acc})

        valid_output,v_pr_vurve = evalueated(model, valid_loader ,triplet_loss)
        v_prcurves.append(v_pr_vurve)
        performances['valid'].append(valid_output)

        test_output,t_pr_vurve = evalueated(model, test_loader ,triplet_loss)
        t_prcurves.append(t_pr_vurve)
        performances['test'].append(test_output)
        
        print(f'Valid Phrase:{gen_output_report(valid_output)},\nTest Phrase:{gen_output_report(test_output)}\n')


    np.save( os.path.join("model_resuls_dir",savepath,'pr_curve',model_name+'PR_Curve.npy') , [v_prcurves]+[t_prcurves])
    with open(os.path.join("model_resuls_dir",savepath,'performance',model_name+'_Performanceinfo.json'),'w') as f:
        json.dump(performances,f)

def run_model_with_earlystopping(model,triplet_loss,optimizer,train_loader,valid_loader,test_loader,model_name,savepath='.',num_epochs=num_epochs,):
    
    logger.info('********______________________________________********')
    logger.info(f'********_________Model {model_name} _________********')
    checkpoint_path = os.path.join("model_resuls_dir",savepath,'models',model_name+'_checkpoint.pth') 
    best_loss = np.inf  
    best_epoch = 0
    v_prcurves = []
    t_prcurves = []
    mkresultdir(savepath)

    logger.info(f'START Training and Testing...\n MODEL {model_name} info :{model}')
    print(f'START Training and Testing...\n MODEL {model_name} info :{model}')

    performances = {"train":[],"valid":[],"test":[]}

    for epoch in range(num_epochs):
        
        train_loss,train_losses,train_acc = train(model, optimizer, train_loader, triplet_loss)
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}] Train Phrase: loss = {train_loss:.4f} ,acc = {train_acc:.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Phrase: loss = {train_loss:.4f} ,acc = {train_acc:.4f}')
        
        performances['train'].append({'AverageLoss':train_loss,'losses':train_losses,'Accuracy':train_acc})

        valid_output,v_pr_vurve = evalueated(model, valid_loader ,triplet_loss)
        v_prcurves.append(v_pr_vurve)
        performances['valid'].append(valid_output)
        print(f'\t\tValid Phrase:{gen_output_report_simple(valid_output)}')
        
        test_output,t_pr_vurve = evalueated(model, test_loader ,triplet_loss)
        t_prcurves.append(t_pr_vurve)
        performances['test'].append(test_output)
        print(f'\t\tTest Phrase:{gen_output_report_simple(test_output)}\n')

        if valid_output['AverageLoss'] < best_loss:
            if epoch ==0: 
                continue
            best_loss = valid_output['AverageLoss']
            best_epoch = epoch
            early_stopping_count = 0
            torch.save({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'train_loss':train_loss,
                        'val_loss': valid_output,
                        'test_loss':test_output
                        },checkpoint_path)
        else:
            early_stopping_count +=1
        if early_stopping_count >4:
            print(f'!!EarlyStopping at epoch {epoch},best epoch {best_epoch} ')
            logger.info(f'EarlyStopping at epoch {epoch},best epoch {best_epoch}')
            break

    np.save( os.path.join("model_resuls_dir",savepath,'pr_curve',model_name+'PR_Curve.npy') , [v_prcurves]+[t_prcurves])
    with open(os.path.join("model_resuls_dir",savepath,'performance',model_name+'_Performanceinfo.json'),'w') as f:
        json.dump(performances,f)


def construct_dataset(triplets):
    import random
    import dataset
    from torch.utils.data import DataLoader

    train_len = int(len(triplets) * train_ratio)
    val_len = int(len(triplets) * val_ratio)


    random.shuffle(triplets)

    train_apn_pairs = triplets[:train_len]
    valid_apn_pairs = triplets[train_len:train_len+val_len]
    test_apn_pairs = triplets[train_len+val_len:]

    train_apn_ASTs = dataset.APCDataset4baseline(train_apn_pairs)

    train_loader = DataLoader(dataset=train_apn_ASTs,
                                batch_size=real_batchsize,
                                shuffle=False,
                                collate_fn=dataset.collate4pairs_fn,
                                drop_last=True)

    valid_apn_ASTs = dataset.APCDataset4baseline(valid_apn_pairs)

    valid_loader = DataLoader(dataset=valid_apn_ASTs,
                                batch_size=real_batchsize,
                                shuffle=False,
                                collate_fn=dataset.collate4pairs_fn,
                                drop_last=True)

    test_apn_ASTs = dataset.APCDataset4baseline(test_apn_pairs)

    test_loader = DataLoader(dataset=test_apn_ASTs,
                                batch_size=real_batchsize,
                                shuffle=False,
                                collate_fn=dataset.collate4pairs_fn,
                                drop_last=True)

    return train_loader,valid_loader,test_loader

def get_dataloader(triplet_num):

    with open('Tripletdataset_92883.json','r') as f:
        alltriplets = json.load(f)

    triplets = alltriplets[:triplet_num]

    return construct_dataset(triplets)

def get_dataloader_binary(triplet_num):
    
    with open('Tripletdataset_Binary_57582.json','r') as f:
        alltriplets = json.load(f)

    triplets = alltriplets[:triplet_num]

    return construct_dataset(triplets)

def get_dataloader_srouce(triplet_num):

    
    with open('Tripletdataset_Source_20418.json','r') as f:
        alltriplets = json.load(f)

    triplets = alltriplets[:triplet_num]

    return construct_dataset(triplets)



def mkresultdir(savedir):
    subdir = ['models','performance','pr_curve']
    for i in subdir:
        path = os.path.join("model_resuls_dir",savedir,i)
        if not os.path.exists(path):
            os.makedirs(path)

