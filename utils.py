import matplotlib.pyplot as plt
import pandas as pd
from imblearn.metrics import classification_report_imbalanced
import seaborn as sns
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def confusion_plot(pred, y_true):
    sns.set(rc={'figure.figsize':(5,4)})
    fault_labels = np.unique(y_true)
    print(fault_labels)
    cm_array = confusion_matrix(y_true, pred,labels=fault_labels)
    df_cm = pd.DataFrame(cm_array, index = fault_labels,
                      columns = fault_labels)
    sns.heatmap(df_cm,annot=True)
    plt.show()
    
    print(classification_report_imbalanced(np.array(y_true), np.array(pred)))
    return plt
    
def plot_confusion_matrix(cm, classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    print(cm)
    plt.figure(figsize=(4, 4)) 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(shrink=0.7)
    tick_marks = np.arange(len(list(range(cm.shape[0]))))
#     plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes,rotation=90)
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt

def plot_pairs(pairs,plot_idx=None):
    nc,w,h = pairs[0].shape[0:3]
    if not plot_idx:
        plot_idx = list(range(nc))
    fig, ax = plt.subplots(nrows=len(plot_idx),ncols=4, figsize=(16, len(plot_idx)))
    for i,v in enumerate(plot_idx):
        ax[i][0].plot(pairs[0][v,:,0,0])
        ax[i][0].get_yaxis().set_visible(False)
        ax[i][0].get_xaxis().set_visible(False)
        ax[i][1].plot(pairs[1][v,:,0,0])
        ax[i][1].get_yaxis().set_visible(False)
        ax[i][1].get_xaxis().set_visible(False)
        ax[i][2].plot(pairs[0][v,:,1,0])
        ax[i][2].get_yaxis().set_visible(False)
        ax[i][2].get_xaxis().set_visible(False)
        ax[i][3].plot(pairs[1][v,:,1,0])
        ax[i][3].get_yaxis().set_visible(False)
        ax[i][3].get_xaxis().set_visible(False)
    plt.show()
    
def noise_rw(x,snr,isplot = False):
    snr1 = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2,axis=0) / len(x)
    npower = xpower / snr1
    noise = np.random.normal(0, np.sqrt(npower), x.shape)
    noise_data=x+noise
    if(isplot):
        print(snr,snr1,npower)
        print(np.sum(noise ** 2)/len(x))
        fig, axs = plt.subplots(nrows=3,ncols=x.shape[1], figsize=(8*x.shape[1], 6))
        for i in range(x.shape[1]):
            axs[0][i].plot(x[:,i])
            axs[0][i].set_title(signal_labels[i] + ' signal')
            axs[0][i].get_xaxis().set_visible(False)
            axs[1][i].plot(noise[:,i])
            axs[1][i].set_title(signal_labels[i] +' noise')
            axs[1][i].get_xaxis().set_visible(False)
            axs[2][i].plot(noise_data[:,i])
            axs[2][i].set_title(signal_labels[i] +' noise signal')
        plt.show()
    return noise_data

def plot_with_labels(data):
    #loop through labels and plot each cluster
    sns.set(rc={'figure.figsize':(5,5)})
    plt.figure()
    for i, label in enumerate(range(10)):

        #add data points 
        plt.scatter(x=data.loc[data['label']==label, 'x'], 
                    y=data.loc[data['label']==label,'y'], 
                    color=cm.rainbow(int(255 * i / 9)), 
                    alpha=0.20)

        #add label
        plt.annotate(label, 
                     data.loc[data['label']==label,['x','y']].mean(),
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=14,
                     weight='bold',
                     color='black') 