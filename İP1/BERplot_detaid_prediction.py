#!/usr/bin/env python
# coding: utf-8

# # This code obtains the predictions of DetAid for BER plot dataset and saves them in csv format for comparison in MATLAB

# In[1]:


import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statistics


# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


# In[2]:


class MultilabelQPMMDataset(Dataset):
    def __init__(self, csvPathFeatures, csvPathLabels):
        
        features = pd.read_csv(csvPathFeatures, header=None)
        labels = pd.read_csv(csvPathLabels, header=None)
        
        data = np.array(features)
        labels = np.array(labels)
        
        self.inputs = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        all_labels = self.labels[idx, :]

        label1 = all_labels[0]
        label2 = all_labels[1]
        label3 = all_labels[2]
        sample = {'inputs': self.inputs[idx,:],
                  'labels': {'s_idx':label1, 'i_idx':label2, 'j_idx':label3}}
        
        return sample


# In[3]:


class IndexClassifierv1(nn.Module):
    def __init__(self, input_size, n_i_idx, n_j_idx):
        super().__init__()
        
        self.Y_VECTOR_SIZE = 40 # complex flattened size
        self.input_sz = self.Y_VECTOR_SIZE
        
        self.model_backbone = nn.Sequential(
            nn.Linear(self.input_sz, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        self.i_class = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, n_i_idx),
            nn.ReLU(),
            nn.Linear(n_i_idx, n_i_idx),
        )
        
        self.j_class = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, n_j_idx),
            nn.ReLU(),
            nn.Linear(n_j_idx, n_j_idx),
        )

    def forward(self, x):
        x = self.model_backbone(x)
        return {
            'i_idx': self.i_class(x),
            'j_idx': self.j_class(x)
        }


# In[4]:


# SNR_VALS = np.linspace(0, 20, 3, dtype='int32')
SNR_VALS = [10]


# In[5]:


S_SIZE = 16
I_SIZE = 16
J_SIZE = 16
FEATURE_VECTOR_SZ = 40
BATCH_SIZE = 128

def get_prediction_df(output):
    predictions_dict = {}
    max_soft_dict = {}
    for pred_type in list(output.keys()):
        max_prob, predicted_labels = torch.max(torch.nn.functional.softmax(outputs[pred_type]), dim=1)
        predicted_labels = predicted_labels.cpu().numpy()
        predictions_dict[pred_type] = predicted_labels
        max_prob = max_prob.cpu().numpy()
        predictions_dict[pred_type + '_prob'] = max_prob
        max_soft_dict[pred_type] = max_prob
    return predictions_dict, max_soft_dict

def get_err_bits_df(labels, predictions, max_soft):
    trueOrNot_dict = {}
    for pred_type in ['i_idx', 'j_idx']:
        curr_predictions = predictions[pred_type].tolist()
        curr_real = labels[pred_type].tolist()
        
        curr_predictions = np.array(curr_predictions)
        curr_real = np.array(curr_real)
        
        trueOrNot_dict['real ' + pred_type] = []
        trueOrNot_dict['real bits ' + pred_type] = []
        trueOrNot_dict['pred bits ' + pred_type] = []
        trueOrNot_dict[pred_type+'_true_prediction'] = []
        trueOrNot_dict['bit error ' + pred_type] = []
        for i, pred in enumerate(curr_predictions):
            real = np.binary_repr(curr_real[i], width = 4)
            prediction = np.binary_repr(pred, width = 4)
            if real != prediction:
                trueOrNot_dict['real ' + pred_type].append(curr_real[i])
                new_err = sum(real[j] != prediction[j] for j in range(len(real)))
                trueOrNot_dict['real bits ' + pred_type].append(real)
                trueOrNot_dict['pred bits ' + pred_type].append(prediction)
                trueOrNot_dict['bit error ' + pred_type].append(new_err)
                trueOrNot_dict[pred_type+'_true_prediction'].append(0)
            else:
                trueOrNot_dict['real ' + pred_type].append(curr_real[i])
                trueOrNot_dict[pred_type+'_true_prediction'].append(1)
                new_err = sum(real[j] != prediction[j] for j in range(len(real)))
                trueOrNot_dict['real bits ' + pred_type].append(real)
                trueOrNot_dict['pred bits ' + pred_type].append(prediction)
                trueOrNot_dict['bit error ' + pred_type].append(new_err)

    return trueOrNot_dict


for currSNR in SNR_VALS:
    MODEL_PATH = f"./savedmodels_ptot4_stable/PaperModel_IndexClassifierv1_{currSNR}"
    LOAD_PATH = f"./savedmodels_ptot4_stable/PaperModel_IndexClassifierv1_{currSNR}"
    print(f'Testing {LOAD_PATH}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')
    
    CSV_PATH_FEATURES = f"features_Nt4_Nr4_M2_SNRdb{currSNR}_modtypePSK.csv"
    CSV_PATH_LABELS = f"labels_Nt4_Nr4_M2_SNRdb{currSNR}_modtypePSK.csv"
    print(f'Running on ./BERPlotData5e5/{CSV_PATH_FEATURES}')
    
    test_data = MultilabelQPMMDataset(f'./BERPlotData5e5/{CSV_PATH_FEATURES}', f'./BERPlotData5e5/{CSV_PATH_LABELS}')
    print(f'Total number of samples in the dataset: {test_data.__len__()}')
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    
    model = IndexClassifierv1(FEATURE_VECTOR_SZ, I_SIZE, J_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))               

    # Evaluation
    model.eval()
    with torch.no_grad():
        all_preds, trueornot = [], []
        for i, data in enumerate(test_dataloader):
            inputs = data['inputs'].to(device)
            labels = data['labels']
            print(labels)
            # Forward pass
            outputs = model(inputs)
            predictions, max_soft = get_prediction_df(outputs)
            all_preds.append(predictions)
    #         if i % 5 == 0:
    #             print_preds(labels, predictions)
            trueOrNot_dict = get_err_bits_df(labels, predictions, max_soft)
            trueornot.append(trueOrNot_dict)
    
    # Save prediction data
    df_all_preds = {}
    for i, dicts in enumerate(all_preds):
        for k in dicts.keys():
            if i == 0:
                df_all_preds['prediction '+k] = dicts[k]
                continue
            df_all_preds['prediction '+k] = np.concatenate((df_all_preds['prediction '+k], dicts[k]))


    df_trueornot = {}
    for i, dicts in enumerate(trueornot):
        for k in dicts.keys():
            if i == 0:
                df_trueornot[k] = dicts[k]
                continue
            df_trueornot[k] = np.concatenate((df_trueornot[k], dicts[k]))


    df_all_preds = pd.DataFrame(df_all_preds)
    df_trueornot = pd.DataFrame(df_trueornot)
    dfs = [df_all_preds, df_trueornot]
    df_overall = pd.concat(dfs, axis=1)

    model_predictions_df = df_overall.loc[:, ['prediction i_idx', 'prediction i_idx_prob', 'prediction j_idx', 'prediction j_idx_prob']]
    model_predictions_df.to_csv(f"DNN_predictions_{currSNR}dB.csv",sep=",",header=False,index=False)


# In[6]:


df_overall


# In[7]:


df_overall[df_overall.loc[:, 'bit error i_idx'] != 0]


# In[8]:


df_overall.describe()


# In[9]:


df_overall['real i_idx'].value_counts()


# In[10]:


sns.histplot(data=df_overall, x="real i_idx", hue="i_idx_true_prediction").set(title='i index Distribution in Test Set')


# In[11]:


sns.histplot(data=df_overall, x="real j_idx", hue="j_idx_true_prediction").set(title='j index Distribution in Test Set')


# In[12]:


plt.figure(figsize=(10,10))
sns.histplot(data=df_overall,
             x="prediction i_idx_prob",
             hue="i_idx_true_prediction",
             binwidth=0.05,
             palette=['#c70000', '#1db954'],
             element='step')
plt.xlabel('Prediction Probability for i Index', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.legend(title='True or False Prediction',
           labels=['True', 'False'],
           loc='upper left',
           fontsize=16)
plt.yscale('log')
plt.title('Frequency of i Index Prediction Probabilities (Color-coded by true or false prediction)')
plt.savefig('i_index_prob_hist.png')


# In[13]:


plt.figure(figsize=(10,10))
sns.histplot(data=df_overall,
             x="prediction j_idx_prob",
             hue="j_idx_true_prediction",
             binwidth=0.05,
             palette=['#c70000', '#1db954'],
             element='bars')
plt.xlabel('Prediction Probability for j Index', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.legend(title='True or False Prediction',
           labels=['True', 'False'],
           loc='upper left',
           fontsize=16)
plt.yscale('log')
plt.title('Frequency of j Index Prediction Probabilities (Color-coded by true or false prediction)')
plt.savefig('j_index_prob_hist.png')


# In[14]:


myplot = sns.kdeplot(data=df_overall, x="prediction i_idx_prob", hue="i_idx_true_prediction")
#myplot.set_ylim(0,1)


# In[15]:


myplot = sns.kdeplot(data=df_overall, x="prediction j_idx_prob", hue="j_idx_true_prediction")
myplot.set_ylim(0,1)


# In[33]:


df_confident = df_overall[(df_overall['prediction i_idx_prob'] >0.95)&(df_overall['prediction j_idx_prob'] > 0.95)]
desc = df_confident.describe()
desc


# In[34]:


i_ber = desc['bit error i_idx']['mean']
j_ber = desc['bit error j_idx']['mean']
predicted_count = desc['bit error i_idx']['count']

i_bit_err = df_confident['bit error i_idx'].sum()
j_bit_err = df_confident['bit error j_idx'].sum()

print('Index-bits BER: ', (i_bit_err + j_bit_err)/(predicted_count*8))
print('Coverage (Ratio of predicted samples) : ', predicted_count/len(df_overall))


# In[35]:


predicted_count


# In[19]:


i_bit_err + j_bit_err


# In[20]:


quantile_val = 0.25
qi = df_overall['prediction i_idx_prob'].quantile(q=quantile_val)
qj = df_overall['prediction j_idx_prob'].quantile(q=quantile_val)
print(f'Quantile {quantile_val}: i -> {qi} j -> {qj}')


# In[ ]:




