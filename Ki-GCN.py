# coding: utf-8
"""
Created on Mon Mar  8 12:10:44 2021

@author: wjh14
"""
# In[1]:
import pandas as pd
import numpy as np
import evaluation as eva
import dgl
import json
import heapq
from collections import Counter
import dgl.nn as dglnn
import torch
import torch.nn as nn
from scipy import interp
from torch.nn.functional import softmax
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold,learning_curve,ShuffleSplit
from sklearn.feature_selection import SelectKBest,chi2,SelectFromModel
from sklearn.metrics import roc_curve,auc,confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression as LR
from operator import truediv
from sklearn import preprocessing   

# In[2]:
help_msg="""
usage: python ki-gcn.py -v=variants_matrix.csv -k=kinship_matrix.csv -l=label.csv -e=edeg.csv -o output
Options and arguments:
-v: The variants matrix, each row represent a sample,and each column represent a variant.
-k: The kinship matrix calculated by ibs algorithm, it should keep one decimal place.
-l: The label of the samples, it should has the same order as variants matrix and the kinship matrix, the label should be binary.
-e: The edge matrix transformed from the kinship matrix, there are two columns, the first column is the order of samples, the second column is the neighbor of samples in the first column, the third column is the corresponding kinship coefficient.
-o: The prefix of the output file, which contains the selected variants matrix, the default is output_kigcn.
"""


if "-help" in param.keys() or "-h" in param.keys():
    print(help_msg)
    
if "-v" not in param.keys():
    print("Parameter v missing!")
    print(help_msg)
    exit()
else:
    nodes_path=[param["-v"]]
    nodes_data = pd.read_csv('nodes_path',header=1,index_col=0)


if "-k" not in param.keys():
    print("Parameter k missing!")
    print(help_msg)
    exit()
else:
    edges_path=param["-e"]
    edges_data=pd.read_csv(edges_path)


if "-l" not in param.keys():
    print("Parameter l missing!")
    print(help_msg)
    exit()
else:
    label_path=param["-l"]
    my_labels = pd.read_csv(label_path)
    
    
if "-e" not in param.keys():
    print("Parameter e missing!")
    print(help_msg)
    exit()
else:
    edges_mat_path=param["-e"]
    edges_data_matric=pd.read_csv(edges_mat_path)


if "-o" not in param.keys():    
    output="output_kigcn"
else:
    output=param["-o"]

y = my_labels
X = nodes_data


# In[3]:


def print_fig(epochs,accuracy_test,accuracy_train):
    x1 = range(0, epochs)
    x2 = range(0, epochs)
    plt.plot(x1,accuracy_test,'o-',label='Test',lw=1, color='r',alpha=0.3)
    plt.plot(x2,accuracy_train,'o-',label='train',lw=1, color='b',alpha=0.3)
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.show()
def plot_shadow(tprs,aucs,snps_number):
    figname = str(snps_number)+'.png'
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = str(snps_number-1)+'SNPs AUC curve'
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(figname, dpi=300)
    plt.show()


# In[4]:


def plot_auc_curve(y_label,y_score,n_classes):
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # plot class==1
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return fpr,tpr,threshold


# In[5]:

    
class RGCN_encoder(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.encoder = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.label = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
    def forward(self, graph, inputs):
        encoder = self.encoder(graph, inputs)
        encoder = {k: F.relu(v) for k, v in encoder.items()}
        label = self.label(graph, encoder)
        pro = F.softmax(label['user'], dim=1)
        return encoder,pro


# In[6]:
    
    
loss_fn = torch.nn.MSELoss()
def train_model(train_idx,test_idx, edge, real_labels, weight,
                l2norm,epochs,lr,hid_fea):
    
    model = RGCN_encoder(n_hetero_features,hid_fea,2,hetero_graph.etypes)
    opt = torch.optim.Adam(model.parameters(),weight_decay=l2norm,lr=lr)
    accuracy_train = []
    accuracy_test = []
    for epoch in range(epochs):
        model.train()
        decoder,pro = model(hetero_graph, node_features)

        #loss function
        edge_train_real = edge.iloc[train_idx,train_idx]
        edge_train_real = torch.autograd.Variable(torch.from_numpy(np.array(edge_train_real)))
        decoder = decoder['user'][train_idx].to(torch.float32)
        relation = np.dot(decoder.detach().numpy(),decoder.T.detach().numpy())
        realtion_input = torch.autograd.Variable(torch.from_numpy(relation))
        
        loss = loss_fn(realtion_input,edge_train_real) + F.cross_entropy(pro[train_idx], real_labels[train_idx],weight = weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        _, predicted_label = torch.max(pro, dim=1)
        correct_test = torch.sum(real_labels[test_idx] == predicted_label[test_idx]).item()
        correct_train = torch.sum(real_labels[train_idx] == predicted_label[train_idx]).item()
        accuracy_test.append(100 * correct_test / (len(test_idx)))
        accuracy_train.append(100 * correct_train / (len(train_idx)))
        
    model.eval()
    _,pro = model(hetero_graph, node_features)
    
    correct = 0
    _, predicted = torch.max(pro,1)

    correct += (predicted[test_idx] == real_labels[test_idx]).sum()
    
    confusion = confusion_matrix(real_labels[test_idx],predicted[test_idx])
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion,axis = 1)
    each_acc = np.nan_to_num(truediv(list_diag,list_raw_sum))
    
    y_label = bi_label[test_idx]
    y_score = pro[test_idx].detach().numpy()
    fpr = dict()
    tpr = dict()
    threshhold = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], threshhold[i]= roc_curve(y_label[:, i], y_score[:, i],drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])
    return model,fpr[1],tpr[1],accuracy_test,accuracy_train,accuracy_test[epochs-1],accuracy_train[epochs-1]


# In[7]:

    
def REF(x,y,min_feature=10):
    my_svc = svm.SVC(kernel='linear', probability=True,class_weight ="balanced",
                  decision_function_shape='ovo')
    rfecv = RFECV(estimator=my_svc, step=1, cv=StratifiedKFold(10),scoring='accuracy',min_features_to_select=min_feature)
    rfecv.fit(x, y)
    plt.figure(figsize=(6, 3), dpi=400)
    plt.xlabel("Number of features selected")
    plt.ylabel("accuracy")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    a = x.columns[rfecv.support_]
    return a


# In[8]:

 
X_select = SelectKBest(chi2, k=200)
X_new = X_select.fit_transform(X, y)
X_index = X_select.get_support(indices=True).tolist()
X_chi = nodes_data.iloc[:,X_index]
X_new = X_chi
n_hetero_features = X_new.shape[1]
hetero_graph = dgl.heterograph({
    ('user', 'rela_11', 'user'): (np.array(edges_data[edges_data.Weight == 1.1].iloc[:,0]), np.array(edges_data[edges_data.Weight == 1.1].iloc[:,1])),   
    ('user', 'rela_12', 'user'): (np.array(edges_data[edges_data.Weight == 1.2].iloc[:,0]), np.array(edges_data[edges_data.Weight == 1.2].iloc[:,1])),
    ('user', 'rela_13', 'user'): (np.array(edges_data[edges_data.Weight == 1.3].iloc[:,0]), np.array(edges_data[edges_data.Weight == 1.3].iloc[:,1])),
    ('user', 'rela_14', 'user'): (np.array(edges_data[edges_data.Weight == 1.4].iloc[:,0]), np.array(edges_data[edges_data.Weight == 1.4].iloc[:,1])),
    ('user', 'rela_15', 'user'): (np.array(edges_data[edges_data.Weight == 1.5].iloc[:,0]), np.array(edges_data[edges_data.Weight == 1.5].iloc[:,1])),
    ('user', 'rela_16', 'user'): (np.array(edges_data[edges_data.Weight == 1.6].iloc[:,0]), np.array(edges_data[edges_data.Weight == 1.6].iloc[:,1])),
    ('user', 'rela_17', 'user'): (np.array(edges_data[edges_data.Weight == 1.7].iloc[:,0]), np.array(edges_data[edges_data.Weight == 1.7].iloc[:,1])),
    ('user', 'rela_18', 'user'): (np.array(edges_data[edges_data.Weight == 1.8].iloc[:,0]), np.array(edges_data[edges_data.Weight == 1.8].iloc[:,1])),
    ('user', 'rela_19', 'user'): (np.array(edges_data[edges_data.Weight == 1.9].iloc[:,0]), np.array(edges_data[edges_data.Weight == 1.9].iloc[:,1])),
    ('user', 'rela_20', 'user'): (np.array(edges_data[edges_data.Weight == 2.0].iloc[:,0]), np.array(edges_data[edges_data.Weight == 2.0].iloc[:,1]))
})
hetero_graph.nodes['user'].data['feature'] = torch.tensor(np.array(X_new))
hetero_graph.nodes['user'].data['label'] = torch.tensor(my_labels.label_40)
user_feats = hetero_graph.nodes['user'].data['feature']
labels = hetero_graph.nodes['user'].data['label']
node_features = {'user': user_feats}
bi_label = label_binarize(labels,classes=[0,1,2])
real_labels = hetero_graph.nodes['user'].data['label']
Counter(real_labels.numpy())


# In[9]:

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
all_acc=[]
for i in range(5):
    print(i)
    skf = StratifiedKFold(n_splits = 5, shuffle=True)
    mean_fpr = np.linspace(0, 1, 100)
    acc_trains = []
    acc_tests = []
    tprs = []
    aucs = []
    mols = []
    epochs = 500
    i = 0
    for fold,(train_idx,test_idx) in enumerate(skf.split(user_feats, labels)):
        model,fpr,tpr,a_list,b_list,acc_test,acc_train = train_model(train_idx,test_idx,edge=edges_data_matric,
                            real_labels = hetero_graph.nodes['user'].data['label'],
                            weight = torch.FloatTensor([1,2]),
                            l2norm = 0.4,epochs = epochs,lr=1e-2,hid_fea=30)
        acc_tests.append(acc_test)
        acc_trains.append(acc_train)
        mols.append(model)

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        i += 1

    all_acc.append(np.mean(acc_tests))
print("total_acc",np.mean(all_acc))
print("total_var",np.var(all_acc))


# In[10]:


weight = []
for i in range(5):
    a = np.abs(mols[i].encoder.state_dict()['mods.rela_11.weight'])+np.abs(mols[i].encoder.state_dict()['mods.rela_12.weight'])+np.abs(mols[i].encoder.state_dict()['mods.rela_13.weight'])+np.abs(mols[i].encoder.state_dict()['mods.rela_14.weight'])+np.abs(mols[i].encoder.state_dict()['mods.rela_15.weight'])+\
        np.abs(mols[i].encoder.state_dict()['mods.rela_16.weight'])+np.abs(mols[i].encoder.state_dict()['mods.rela_17.weight'])+np.abs(mols[i].encoder.state_dict()['mods.rela_18.weight'])+np.abs(mols[i].encoder.state_dict()['mods.rela_19.weight'])+np.abs(mols[i].encoder.state_dict()['mods.rela_20.weight'])

    b = np.abs(mols[i].label.state_dict()['mods.rela_11.weight'])+np.abs(mols[i].label.state_dict()['mods.rela_12.weight'])+np.abs(mols[i].label.state_dict()['mods.rela_13.weight'])+np.abs(mols[i].label.state_dict()['mods.rela_14.weight'])+np.abs(mols[i].label.state_dict()['mods.rela_15.weight'])+\
        np.abs(mols[i].label.state_dict()['mods.rela_16.weight'])+np.abs(mols[i].label.state_dict()['mods.rela_17.weight'])+np.abs(mols[i].label.state_dict()['mods.rela_18.weight'])+np.abs(mols[i].label.state_dict()['mods.rela_19.weight'])+np.abs(mols[i].label.state_dict()['mods.rela_20.weight'])

    weight.append(np.dot(a.detach().numpy(),b.detach().numpy())[:,1])

index = []
for i in range(0,5):
    ind = list(map(weight[i].tolist().index, heapq.nlargest(50, weight[i])))
    index = index +ind

b = pd.value_counts(index).rename_axis('unique_values').reset_index(name='counts')
b4 = b[b['counts']>=2].unique_values.tolist()
print(len(b4))
kigcn_snp_filter = X_chi.iloc[:,b4]
kigcn_snp_out=REF(x,y)
print(kigcn_snp_out.shape)
kigcn_snp_out.to_csv(output+'.csv')


