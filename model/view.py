#python3

import torch
import os,json
import numpy as np
import matplotlib.pyplot as plt
from Hyperparameter import *

def get_report(modelname,modoldir='.'):

    with open(os.path.join("model_resuls_dir",modoldir,"performance",modelname+'_Performanceinfo.json') ,'r') as fp:
        return json.load(fp)
    
def get_model(modelname,modoldir='.'):
    
    return torch.load(os.path.join("model_resuls_dir",modoldir,"models",modelname+'_checkpoint.pth'))

def get_smooth_value(valuelist,smoothing_window):
  
    smoothed = []
    value_history = []
    for i in valuelist:
        value_history.append(i)

   
        if len(value_history) >= smoothing_window:
            smoothed_value = np.mean(value_history[-smoothing_window:])
            smoothed.append(smoothed_value)
        else:
            smoothed_value = np.mean(value_history)
            smoothed.append(smoothed_value)
    return smoothed

def get_losses(perform,modelname,smoothing=None):


    train_loss = [ i['AverageLoss'] for i in perform['train']]
    valid_loss = [ i['AverageLoss'] for i in perform['valid']]
    test_loss = [ i['AverageLoss'] for i in perform['test']]

    if smoothing is not None:
        train_loss = get_smooth_value(train_loss,smoothing)
        valid_loss = get_smooth_value(valid_loss,smoothing)
        test_loss = get_smooth_value(test_loss,smoothing)

    plt.figure(figsize=(4,3))

    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss,  label='train')
    plt.plot(epochs, valid_loss,  label='valid')
    plt.plot(epochs, test_loss, label='test')
    plt.title(modelname +' Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def acc_compare(modelperformance_dic,phase,smooth=None):
   
    for modelname,modelperform in modelperformance_dic.items():
        if smooth is not None:
            model_acc = get_smooth_value([  i['Accuracy']  for i in modelperform[phase]] ,smooth)
        else:
            model_acc = [  i['Accuracy'] for i in modelperform[phase]]
        epochs = range(1, len(model_acc) + 1)
        plt.plot(epochs, model_acc,  label=modelname)

    plt.title('Accuracy Curve of Baseline')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

def loss_compare(modelperformance_dic,phase,smooth=None):
   
    for modelname,modelperform in modelperformance_dic.items():
        if smooth is not None:
            model_loss = get_smooth_value([ i['AverageLoss'] for i in modelperform[phase]] ,smooth)
        else:
            model_loss = [ i['AverageLoss'] for i in modelperform[phase]]
        epochs = range(1, len(model_loss) + 1)
        plt.plot(epochs, model_loss,  label=modelname)

    plt.title(phase +' Loss Curve of Baseline')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

def get_best_epoch_prinfo(modelresults,perform):
    
    best_epoch = modelresults['epoch']

    pcs = perform['test'][best_epoch]['matrix']['PrecisionMatrix']
    rcl = perform['test'][best_epoch]['matrix']['RecalMatrix']
    prauc = perform['test'][best_epoch]['PR_AUC']

    return np.array(pcs),np.array(rcl),prauc

def plot_pr_curves_compare(modelperformance_dic,model_dict):


    for modelname,modelperform in modelperformance_dic.items():
        precision, recall, auc_score = get_best_epoch_prinfo(model_dict[modelname],modelperform)
        plt.plot(recall, precision, label='{}    (PR_AUC = {:0.3f})'.format(modelname, auc_score))

   
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Precision-Recall Curve for Baseline models')

    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()