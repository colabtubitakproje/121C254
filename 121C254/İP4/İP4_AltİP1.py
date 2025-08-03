# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import itertools
import math
from math import floor, log2, sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from torchmetrics.functional import precision_recall


class PHYnet(nn.Module):

    def __init__(self):
        super(PHYnet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(572, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PHYDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data).float()
        # self.targets = F.one_hot(torch.LongTensor(targets)).float()
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.data)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def train(model, train_loader, val_loader, crit, optimizer, epochs):
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
    print("\n Training starts")
    last_loss = 100
    triggertimes = 0
    for e in (range(1, epochs + 1)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = crit(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                y_val_pred = model(X_val_batch)

                val_loss = crit(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
            # Early stopping
            current_loss = val_epoch_loss
            if current_loss > last_loss:
                triggertimes += 1
                if triggertimes >= patience:
                    print('Early stopping!')
                    break
            last_loss = current_loss

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        print(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')


def test(model1, model2, model3, model4, model5, loader):
    ############## Testing
    test_epoch_acc = 0
    correct_detection = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # evaluate the model on the test set
            y_test_pred_tx1 = model1(inputs)
            y_test_pred_tx2 = model2(inputs)
            y_test_pred_tx3 = model3(inputs)
            y_test_pred_tx4 = model4(inputs)
            y_test_pred_tx5 = model5(inputs)

            y_pred_softmax1 = torch.log_softmax(y_test_pred_tx1, dim=1)
            _, y_pred_tags1 = torch.max(y_pred_softmax1, dim=1)

            y_pred_softmax2 = torch.log_softmax(y_test_pred_tx2, dim=1)
            _, y_pred_tags2 = torch.max(y_pred_softmax2, dim=1)

            y_pred_softmax3 = torch.log_softmax(y_test_pred_tx3, dim=1)
            _, y_pred_tags3 = torch.max(y_pred_softmax3, dim=1)

            y_pred_softmax4 = torch.log_softmax(y_test_pred_tx4, dim=1)
            _, y_pred_tags4 = torch.max(y_pred_softmax4, dim=1)

            y_pred_softmax5 = torch.log_softmax(y_test_pred_tx5, dim=1)
            _, y_pred_tags5 = torch.max(y_pred_softmax5, dim=1)

            y_pred_tags = torch.vstack((y_pred_tags1, y_pred_tags2, y_pred_tags3, y_pred_tags4, y_pred_tags5)).sum(axis=0)
            y_pred_tags[y_pred_tags < 5] = 0
            targets[targets < 5] = 0
            correct_pred = (y_pred_tags == targets).float()

            TP = TP + torch.sum(torch.logical_and(y_pred_tags == 5, targets == 5)).item()
            TN = TN + torch.sum(torch.logical_and(y_pred_tags == 0, targets == 0)).item()
            FP = FP + torch.sum(torch.logical_and(y_pred_tags == 5, targets == 0)).item()
            FN = FN + torch.sum(torch.logical_and(y_pred_tags == 0, targets == 5)).item()

            acc = correct_pred.sum() / len(correct_pred)
            acc = torch.round(acc * 100)
            test_epoch_acc += acc.item()
        print("Deep Learning")
        print("T+ = ", TP)
        print("T- = ", TN)
        print("F+ = ", FP)
        print("F- = ", FN)
        print("Precision", TP / (TP + FP))
        print("Recall", TP / (TP + FN))
        print("Accuracy", test_epoch_acc / len(test_loader))
        print("")


def test_traditional(features, data, W):
    tx1_traditional = features[0].mean(axis=1)
    tx2_traditional = features[1].mean(axis=1)
    tx3_traditional = features[2].mean(axis=1)
    tx4_traditional = features[3].mean(axis=1)
    tx5_traditional = features[4].mean(axis=1)
    tx_test_traditional = data.mean(axis=1)

    tx_test_traditional[np.where((tx1_traditional.mean() - tx1_traditional.std() * W < tx_test_traditional) &
                                 (tx1_traditional.mean() + tx1_traditional.std() * W > tx_test_traditional))] = 0
    tx_test_traditional[np.where((tx2_traditional.mean() - tx2_traditional.std() * W < tx_test_traditional) &
                                 (tx2_traditional.mean() + tx2_traditional.std() * W > tx_test_traditional))] = 0
    tx_test_traditional[np.where((tx3_traditional.mean() - tx3_traditional.std() * W < tx_test_traditional) &
                                 (tx3_traditional.mean() + tx3_traditional.std() * W > tx_test_traditional))] = 0
    tx_test_traditional[np.where((tx4_traditional.mean() - tx4_traditional.std() * W < tx_test_traditional) &
                                 (tx4_traditional.mean() + tx4_traditional.std() * W > tx_test_traditional))] = 0
    tx_test_traditional[np.where((tx5_traditional.mean() - tx5_traditional.std() * W < tx_test_traditional) &
                                 (tx5_traditional.mean() + tx5_traditional.std() * W > tx_test_traditional))] = 0

    tx_test_traditional[tx_test_traditional > 0] = 5
    targets_test[targets_test < 5] = 0

    correct_pred_traditional = (tx_test_traditional == targets_test)
    acc_traditional = correct_pred_traditional.sum() / len(correct_pred_traditional)
    acc_traditional = (acc_traditional * 100)

    TP_traditional = np.logical_and(tx_test_traditional == 5, targets_test == 5).sum()
    TN_traditional = np.logical_and(tx_test_traditional == 0, targets_test == 0).sum()
    FP_traditional = np.logical_and(tx_test_traditional == 5, targets_test == 0).sum()
    FN_traditional = np.logical_and(tx_test_traditional == 0, targets_test == 5).sum()

    print("Traditional Test")
    print("T+ = ", TP_traditional)
    print("T- = ", TN_traditional)
    print("F+ = ", FP_traditional)
    print("F- = ", FN_traditional)
    print("Precision", TP_traditional / (TP_traditional + FP_traditional))
    print("Recall", TP_traditional / (TP_traditional + FN_traditional))
    print("Accuracy", acc_traditional)
    print("")


def find_best_stdw(features, data):
    for i in np.arange(0.1, 3.5, 0.1):
        test_traditional(features, data, i)


#numSym = 200  # number of OFDM Symbols to transmit
#EbNot_dB = 4  # bit to noise ratio
#EbNo_lin = 10 ** (EbNot_dB / 10)
#M = 2  # modulation order (2,4,8,16 etc.)
#N = 256  # FFT size or total number of subcarriers
#Ncp = 30  # number of symbols allocated to cyclic prefix
#n = 4  # number of subcarriers in a subblock
#k = 2  # number of active subcarriers in a subblock
#v = 6  # channel order
#ICSI = 0  # 1--> imperfect CSI, 0--> perfect CSI
#rho = 0  # jamming parameter
#L = v - 1  # channel order - 1, number of side paths
#SJR_dB = 10  # signal to jamming ratio
#dist = [3, 4, 5, 6, 7]  # node distances
#PLexp = [2, 3, 3, 3, 3]  # node path loss exponents
##########################
stdW = 1/10
EPOCHS = 30
BATCH_SIZE = 1000
patience = 5
###########################

file_path = '9_5 2 3_5 8 5_5 7.txt'
sys.stdout = open(file_path, "w")

###########################  Train data
with open('features_train.npy', 'rb') as f:
    features_train = np.load(f)
with open('targets_train.npy', 'rb') as f:
    targets_train = np.load(f)
with open('features_val.npy', 'rb') as f:
    features_val = np.load(f)
with open('targets_val.npy', 'rb') as f:
    targets_val = np.load(f)
with open('features_test.npy', 'rb') as f:
    features_test = np.load(f)
with open('targets_test.npy', 'rb') as f:
    targets_test = np.load(f)


data_train = features_train.reshape(-1, features_train.shape[-1])
data_val = features_val.reshape(-1, features_val.shape[-1])
data_test = features_test.reshape(-1, features_test.shape[-1])

data_train = np.square(data_train)
data_val = np.square(data_val)
data_test = np.square(data_test)


##########################
model_1 = PHYnet()
model_2 = PHYnet()
model_3 = PHYnet()
model_4 = PHYnet()
model_5 = PHYnet()

################
dataset_train_tx1 = PHYDataset(data_train, targets_train[0])
dataset_val_tx1 = PHYDataset(data_val, targets_val[0])

dataset_train_tx2 = PHYDataset(data_train, targets_train[1])
dataset_val_tx2 = PHYDataset(data_val, targets_val[1])

dataset_train_tx3 = PHYDataset(data_train, targets_train[2])
dataset_val_tx3 = PHYDataset(data_val, targets_val[2])

dataset_train_tx4 = PHYDataset(data_train, targets_train[3])
dataset_val_tx4 = PHYDataset(data_val, targets_val[3])

dataset_train_tx5 = PHYDataset(data_train, targets_train[4])
dataset_val_tx5 = PHYDataset(data_val, targets_val[4])

dataset_test = PHYDataset(data_test, targets_test)

train_loader_tx1 = DataLoader(dataset=dataset_train_tx1, batch_size=BATCH_SIZE, shuffle=True)
val_loader_tx1 = DataLoader(dataset=dataset_val_tx1, batch_size=BATCH_SIZE, shuffle=True)

train_loader_tx2 = DataLoader(dataset=dataset_train_tx2, batch_size=BATCH_SIZE, shuffle=True)
val_loader_tx2 = DataLoader(dataset=dataset_val_tx2, batch_size=BATCH_SIZE, shuffle=True)

train_loader_tx3 = DataLoader(dataset=dataset_train_tx3, batch_size=BATCH_SIZE, shuffle=True)
val_loader_tx3 = DataLoader(dataset=dataset_val_tx3, batch_size=BATCH_SIZE, shuffle=True)

train_loader_tx4 = DataLoader(dataset=dataset_train_tx4, batch_size=BATCH_SIZE, shuffle=True)
val_loader_tx4 = DataLoader(dataset=dataset_val_tx4, batch_size=BATCH_SIZE, shuffle=True)

train_loader_tx5 = DataLoader(dataset=dataset_train_tx5, batch_size=BATCH_SIZE, shuffle=True)
val_loader_tx5 = DataLoader(dataset=dataset_val_tx5, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True)

optimizer1 = optim.Adam(model_1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model_2.parameters(), lr=0.001)
optimizer3 = optim.Adam(model_3.parameters(), lr=0.001)
optimizer4 = optim.Adam(model_4.parameters(), lr=0.001)
optimizer5 = optim.Adam(model_5.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
# criterion = nn.CTCLoss()

train(model_1, train_loader_tx1, val_loader_tx1, criterion, optimizer1, EPOCHS)
train(model_2, train_loader_tx2, val_loader_tx2, criterion, optimizer2, EPOCHS)
train(model_3, train_loader_tx3, val_loader_tx3, criterion, optimizer3, EPOCHS)
train(model_4, train_loader_tx4, val_loader_tx4, criterion, optimizer4, EPOCHS)
train(model_5, train_loader_tx5, val_loader_tx5, criterion, optimizer5, EPOCHS)

test(model_1, model_2, model_3, model_4, model_5, test_loader)

print("Traditional with squaring data")
find_best_stdw(features_train, data_test)

#print("Traditional without squaring data")
#find_best_stdw(features_train, data_test)

#test_traditional(features_train, data_test, stdW)

"""
############## Model 1
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}
print("\n Model 1")
last_loss = 100
triggertimes = 0
for e in (range(1, EPOCHS + 1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model1.train()
    for X_train_batch, y_train_batch in train_loader_tx1:
        optimizer1.zero_grad()

        y_train_pred = model1(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)

        train_loss.backward()
        optimizer1.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    # VALIDATION
    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_acc = 0

        model1.eval()
        for X_val_batch, y_val_batch in val_loader_tx1:
            y_val_pred = model1(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
        # Early stopping
        current_loss = val_epoch_loss
        if current_loss > last_loss:
            triggertimes += 1
            if triggertimes >= patience:
                print('Early stopping!')
                break
        last_loss = current_loss

    loss_stats['train'].append(train_epoch_loss / len(train_loader_tx1))
    loss_stats['val'].append(val_epoch_loss / len(val_loader_tx1))
    accuracy_stats['train'].append(train_epoch_acc / len(train_loader_tx1))
    accuracy_stats['val'].append(val_epoch_acc / len(val_loader_tx1))

    print(
        f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader_tx1):.5f} | Val Loss: {val_epoch_loss / len(val_loader_tx1):.5f} | Train Acc: {train_epoch_acc / len(train_loader_tx1):.3f}| Val Acc: {val_epoch_acc / len(val_loader_tx1):.3f}')

############## Model 2
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}
print("\n Model 2")
last_loss = 100
triggertimes = 0
for e in (range(1, EPOCHS + 1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model2.train()

    for X_train_batch, y_train_batch in train_loader_tx2:
        optimizer2.zero_grad()

        y_train_pred = model2(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)

        train_loss.backward()
        optimizer2.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    # VALIDATION
    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_acc = 0

        model2.eval()
        for X_val_batch, y_val_batch in val_loader_tx2:
            y_val_pred = model2(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
        # Early stopping
        current_loss = val_epoch_loss
        if current_loss > last_loss:
            triggertimes += 1
            if triggertimes >= patience:
                print('Early stopping!')
                break
        last_loss = current_loss

    loss_stats['train'].append(train_epoch_loss / len(train_loader_tx2))
    loss_stats['val'].append(val_epoch_loss / len(val_loader_tx2))
    accuracy_stats['train'].append(train_epoch_acc / len(train_loader_tx2))
    accuracy_stats['val'].append(val_epoch_acc / len(val_loader_tx2))

    print(
        f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader_tx2):.5f} | Val Loss: {val_epoch_loss / len(val_loader_tx2):.5f} | Train Acc: {train_epoch_acc / len(train_loader_tx2):.3f}| Val Acc: {val_epoch_acc / len(val_loader_tx2):.3f}')

############## Model 3
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}
print("\n Model 3")
last_loss = 100
triggertimes = 0
for e in (range(1, EPOCHS + 1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model3.train()

    for X_train_batch, y_train_batch in train_loader_tx3:
        optimizer3.zero_grad()

        y_train_pred = model3(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)

        train_loss.backward()
        optimizer3.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    # VALIDATION
    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_acc = 0

        model3.eval()
        for X_val_batch, y_val_batch in val_loader_tx3:
            y_val_pred = model3(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
        # Early stopping
        current_loss = val_epoch_loss
        if current_loss > last_loss:
            triggertimes += 1
            if triggertimes >= patience:
                print('Early stopping!')
                break
        last_loss = current_loss

    loss_stats['train'].append(train_epoch_loss / len(train_loader_tx3))
    loss_stats['val'].append(val_epoch_loss / len(val_loader_tx3))
    accuracy_stats['train'].append(train_epoch_acc / len(train_loader_tx3))
    accuracy_stats['val'].append(val_epoch_acc / len(val_loader_tx3))

    print(
        f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader_tx3):.5f} | Val Loss: {val_epoch_loss / len(val_loader_tx3):.5f} | Train Acc: {train_epoch_acc / len(train_loader_tx3):.3f}| Val Acc: {val_epoch_acc / len(val_loader_tx3):.3f}')

############## Model 4
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}
print("\n Model 4")
last_loss = 100
triggertimes = 0
for e in (range(1, EPOCHS + 1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model4.train()

    for X_train_batch, y_train_batch in train_loader_tx4:
        optimizer4.zero_grad()

        y_train_pred = model4(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)

        train_loss.backward()
        optimizer4.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    # VALIDATION
    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_acc = 0

        model4.eval()
        for X_val_batch, y_val_batch in val_loader_tx4:
            y_val_pred = model4(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
        # Early stopping
        current_loss = val_epoch_loss
        if current_loss > last_loss:
            triggertimes += 1
            if triggertimes >= patience:
                print('Early stopping!')
                break
        last_loss = current_loss

    loss_stats['train'].append(train_epoch_loss / len(train_loader_tx4))
    loss_stats['val'].append(val_epoch_loss / len(val_loader_tx4))
    accuracy_stats['train'].append(train_epoch_acc / len(train_loader_tx4))
    accuracy_stats['val'].append(val_epoch_acc / len(val_loader_tx4))

    print(
        f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader_tx4):.5f} | Val Loss: {val_epoch_loss / len(val_loader_tx4):.5f} | Train Acc: {train_epoch_acc / len(train_loader_tx4):.3f}| Val Acc: {val_epoch_acc / len(val_loader_tx4):.3f}')

############## Model 5
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}
print("\n Model 5")
last_loss = 100
triggertimes = 0
for e in (range(1, EPOCHS + 1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model5.train()

    for X_train_batch, y_train_batch in train_loader_tx5:
        optimizer5.zero_grad()

        y_train_pred = model5(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)

        train_loss.backward()
        optimizer5.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    # VALIDATION
    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_acc = 0

        model5.eval()
        for X_val_batch, y_val_batch in val_loader_tx5:
            y_val_pred = model5(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
        # Early stopping
        current_loss = val_epoch_loss
        if current_loss > last_loss:
            triggertimes += 1
            if triggertimes >= patience:
                print('Early stopping!')
                break
        last_loss = current_loss

    loss_stats['train'].append(train_epoch_loss / len(train_loader_tx5))
    loss_stats['val'].append(val_epoch_loss / len(val_loader_tx5))
    accuracy_stats['train'].append(train_epoch_acc / len(train_loader_tx5))
    accuracy_stats['val'].append(val_epoch_acc / len(val_loader_tx5))

    print(
        f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader_tx5):.5f} | Val Loss: {val_epoch_loss / len(val_loader_tx5):.5f} | Train Acc: {train_epoch_acc / len(train_loader_tx5):.3f}| Val Acc: {val_epoch_acc / len(val_loader_tx5):.3f}')

############## Testing
test_epoch_acc = 0
correct_detection = 0
TP = 0
TN = 0
FP = 0
FN = 0

for i, (inputs, targets) in enumerate(test_loader):
    # evaluate the model on the test set
    y_test_pred_tx1 = model1(inputs)
    y_test_pred_tx2 = model2(inputs)
    y_test_pred_tx3 = model3(inputs)
    y_test_pred_tx4 = model4(inputs)
    y_test_pred_tx5 = model5(inputs)

    y_pred_softmax1 = torch.log_softmax(y_test_pred_tx1, dim=1)
    _, y_pred_tags1 = torch.max(y_pred_softmax1, dim=1)

    y_pred_softmax2 = torch.log_softmax(y_test_pred_tx2, dim=1)
    _, y_pred_tags2 = torch.max(y_pred_softmax2, dim=1)

    y_pred_softmax3 = torch.log_softmax(y_test_pred_tx3, dim=1)
    _, y_pred_tags3 = torch.max(y_pred_softmax3, dim=1)

    y_pred_softmax4 = torch.log_softmax(y_test_pred_tx4, dim=1)
    _, y_pred_tags4 = torch.max(y_pred_softmax4, dim=1)

    y_pred_softmax5 = torch.log_softmax(y_test_pred_tx5, dim=1)
    _, y_pred_tags5 = torch.max(y_pred_softmax5, dim=1)

    y_pred_tags = torch.vstack((y_pred_tags1, y_pred_tags2, y_pred_tags3, y_pred_tags4, y_pred_tags5)).sum(axis=0)
    y_pred_tags[y_pred_tags < 5] = 0
    targets[targets < 5] = 0
    correct_pred = (y_pred_tags == targets).float()
    detection = torch.sum(y_pred_tags == 5).item()

    TP = TP + torch.sum(torch.logical_and(y_pred_tags == 5, targets == 5)).item()
    TN = TN + torch.sum(torch.logical_and(y_pred_tags == 0, targets == 0)).item()
    FP = FP + torch.sum(torch.logical_and(y_pred_tags == 5, targets == 0)).item()
    FN = FN + torch.sum(torch.logical_and(y_pred_tags == 0, targets == 5)).item()


    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    test_epoch_acc += acc.item()
print("Deep Learning")
print("T+ = ", TP)
print("T- = ", TN)
print("F+ = ", FP)
print("F- = ", FN)
print("Precision", TP / (TP + FP))
print("Recall", TP / (TP + FN))
print("Accuracy", test_epoch_acc / len(test_loader))
print("")

tx1_traditional = features_train[0].mean(axis=1)
tx2_traditional = features_train[1].mean(axis=1)
tx3_traditional = features_train[2].mean(axis=1)
tx4_traditional = features_train[3].mean(axis=1)
tx5_traditional = features_train[4].mean(axis=1)
tx_test_traditional = data_test.mean(axis=1)

tx_test_traditional[np.where((tx1_traditional.mean() - tx1_traditional.std() * stdW < tx_test_traditional) &
                             (tx1_traditional.mean() + tx1_traditional.std() * stdW > tx_test_traditional))] = 0
tx_test_traditional[np.where((tx2_traditional.mean() - tx2_traditional.std() * stdW < tx_test_traditional) &
                             (tx2_traditional.mean() + tx2_traditional.std() * stdW > tx_test_traditional))] = 0
tx_test_traditional[np.where((tx3_traditional.mean() - tx3_traditional.std() * stdW < tx_test_traditional) &
                             (tx3_traditional.mean() + tx3_traditional.std() * stdW > tx_test_traditional))] = 0
tx_test_traditional[np.where((tx4_traditional.mean() - tx4_traditional.std() * stdW < tx_test_traditional) &
                             (tx4_traditional.mean() + tx4_traditional.std() * stdW > tx_test_traditional))] = 0
tx_test_traditional[np.where((tx5_traditional.mean() - tx5_traditional.std() * stdW < tx_test_traditional) &
                             (tx5_traditional.mean() + tx5_traditional.std() * stdW > tx_test_traditional))] = 0

tx_test_traditional[tx_test_traditional > 0] = 5
targets_test[targets_test < 5] = 0

correct_pred_traditional = (tx_test_traditional == targets_test)
acc_traditional = correct_pred_traditional.sum() / len(correct_pred_traditional)
acc_traditional = (acc_traditional * 100)

TP_traditional = np.logical_and(tx_test_traditional == 5, targets_test == 5).sum()
TN_traditional = np.logical_and(tx_test_traditional == 0, targets_test == 0).sum()
FP_traditional = np.logical_and(tx_test_traditional == 5, targets_test == 0).sum()
FN_traditional = np.logical_and(tx_test_traditional == 0, targets_test == 5).sum()


print("Traditional Test")
print("T+ = ", TP_traditional)
print("T- = ", TN_traditional)
print("F+ = ", FP_traditional)
print("F- = ", FN_traditional)
print("Precision", TP_traditional / (TP_traditional + FP_traditional))
print("Recall", TP_traditional / (TP_traditional + FN_traditional))
print("Accuracy", acc_traditional)
print("")
"""
print("hello")
