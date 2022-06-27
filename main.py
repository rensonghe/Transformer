#%%
import numpy as np
import pandas as pd
import torch
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import causal_convolution_layer
import Dataloader_revise
# import math
from Dataloader_revise import *
# from epoch import *
from torch.utils.data import DataLoader
# from parallel import DataParallelModel, DataParallelCriterion
 #%%
import torch.nn.functional as F
# device = torch.device("cuda")
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#%%
# model class
class TransformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper
    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel
    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)
    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector
    """

    def __init__(self):
        super(TransformerTimeSeries, self).__init__()
        self.input_embedding = causal_convolution_layer.context_embedding(2, 256, 9)  #定义conv1函数的是图像卷积函数：输入为2个频道，,输出为256张特征图, 卷积核为9*9 正方形
        self.positional_embedding = torch.nn.Embedding(20000, 256)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)

        self.fc1 = torch.nn.Linear(256, 1)
        self.drop_path_prob = 0.0


    def forward(self, x, y, attention_masks):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        z = torch.cat((y.unsqueeze(1),x.unsqueeze(1)),1)
        # print(z.shape,'zzzzzzzz')
        print(y.unsqueeze(1).shape, 'yyyyyyyyyy')
        print(x.unsqueeze(1).shape, 'xxxxxxxxx')
        print(attention_masks.shape,'sssssssss')

        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)


        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1, 0, 2)

        input_embedding = z_embedding + positional_embeddings

        transformer_embedding = self.transformer_decoder(input_embedding, attention_masks)
        output = self.fc1(transformer_embedding.permute(1, 0, 2))
        print(output.shape, 'shape')

        return output

#%%
# 数据加载
data = pd.read_csv('data_2min_bar_2020_2021_new.csv', index_col='time')
data = data.drop(['buyratio','sellratio','tradingprice'],axis=1)

data = data[~data['close'].isin([0])]
#%%

scaler= StandardScaler().fit(data)
# 保存归一化模型
pickle.dump(scaler, open('scale_model.pth', 'wb'))
print('Scaler model saved to {}'.format('scaler_model.pth'))

#%%
train_data = data[(data.index>='2020-01-08 0:00:00')& (data.index<='2020-02-01 0:00:00')]
test_data = data[(data.index>='2021-10-01 0:00:00')& (data.index<='2021-11-10 23:59:59')]


train = pd.DataFrame(scaler.transform(train_data))
test = pd.DataFrame(scaler.transform(test_data))


train_dataset = time_series_decoder_paper(train)  #last output for train data
test_dataset = time_series_decoder_paper(test)

#%%
# model part
def Dp(y_pred,y_true,q):
    return max([q*(y_pred-y_true),(q-1)*(y_pred-y_true)])

def Rp_num_den(y_preds,y_trues,q):
    numerator = np.sum([Dp(y_pred,y_true,q) for y_pred,y_true in zip(y_preds,y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator,denominator

#%%
# from parallel import BalancedDataParallel
# from torch import nn
#
# batch_szie = 32
# gpu0_bsz = 8
# acc_grad = 2
#%%
from torch import nn
# device = torch.device('cuda')
model = TransformerTimeSeries()
#%%
model = nn.DataParallel(model)
model = model.cuda()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#
#     model = BalancedDataParallel(gpu0_bsz // acc_grad, model, dim=0).cuda()
# model.to(device)
# model = BalancedDataParallel(gpu0_bsz // acc_grad, model, dim=0).cuda()



#%%
train_dl = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_dl = DataLoader(test_dataset, batch_size=256)


#%%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = TransformerTimeSeries().to(device)
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
# model.to(device)

#%%
# lr = 0.0005  # learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 20
# criterion = torch.nn.MSELoss()

#%%
t0=7385
future=7
# def train_epoch(model, train_dl, t0, future):
model.train()
train_loss = 0
n = 0
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
#%%

for step, (x, y, attention_masks) in enumerate(train_dl):
    optimizer.zero_grad()

    # model = DataParallelModel(model)
    # parallel_loss = DataParallelCriterion(criterion)

    output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())
    loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + future - 1)], y[:, t0:].cuda())  # not missing data
    # loss = criterion(output.squeeze()[:,(t0-1-10):(t0+24-1-10)],y.cuda()[:,(t0-10):]) # missing data
    loss.backward()
    optimizer.step()

    train_loss += (loss.detach().cpu().item() * x.shape[0])
    n += x.shape[0]

#%%
def train_epoch(model, train_dl, t0, future):
    model.train()
    train_loss = 0
    n = 0
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for step, (x, y, attention_masks) in enumerate(train_dl):

        optimizer.zero_grad()
        output = model(x.to(device), y.to(device), attention_masks[0].to(device))
        loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + future - 1)], y.to(device)[:, t0:])  # not missing data
        # loss = criterion(output.squeeze()[:,(t0-1-10):(t0+24-1-10)],y.cuda()[:,(t0-10):]) # missing data
        loss.backward()
        optimizer.step()

        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]
    return train_loss / n

#%%
def test_epoch(model, test_dl, t0, future):
    with torch.no_grad():
        predictions = []
        observations = []

        model.eval()
        for step, (x, y, attention_masks) in enumerate(test_dl):
            output = model(x.to(device), y.to(device), attention_masks[0].to(device))

            for p, o in zip(output.squeeze()[:, (t0 - 1):(t0 + future - 1)].cpu().numpy().tolist(),
                            y[:, t0:].cpu().numpy().tolist()):  # not missing data
                # for p,o in zip(output.squeeze()[:,(t0-1-10):(t0+24-1-10)].cpu().numpy().tolist(),y.cuda()[:,(t0-10):].cpu().numpy().tolist()): # missing data

                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den

    return Rp
#
#
#
#
#
#
#%%
train_epoch_loss = []
test_epoch_loss = []
Rp_best = 1e5
epochs = 20
model_save_path = 'ConvTransformer_nologsparse.pth'
for e, epoch in enumerate(range(epochs)):
    train_loss = []
    test_loss = []

    l_t = train_epoch(model, train_dl, t0=2643, future=3)
    train_loss.append(l_t)

    Rp = test_epoch(model, test_dl, t0=1772, future=2)

    if Rp_best > Rp:
        Rp_best = Rp
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': Rp,
        # }, model_save_path)

    train_epoch_loss.append(np.mean(train_loss))
    test_epoch_loss.append(np.mean(test_loss))

    print("Epoch {}: Train loss: {} \t Validation loss: {} \t R_p={}".format(e,
                                                                             np.mean(train_loss),
                                                                             np.mean(test_loss), Rp))

    print("Rp best={}".format(Rp_best))



#%%
# def eval_epoch(model, val_dl, t0, future):
#     model.eval()
#     eval_loss = 0
#     n = 0
#     with torch.no_grad():
#         for step, (x, y, attention_masks) in enumerate(val_dl):
#             output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())
#             loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + 24 - 1)], y.cuda()[:, t0:])  # not missing data
#             # loss = criterion(output.squeeze()[:,(t0-1-10):(t0+24-1-10)],y.cuda()[:,(t0-10):]) # missing data
#
#             eval_loss += (loss.detach().cpu().item() * x.shape[0])
#             n += x.shape[0]
#
#     return eval_loss / n




# criterion = torch.nn.MSELoss()
#
# train_dl = DataLoader(train_data,batch_size=32,shuffle=True)
# val_dl = DataLoader(val_data,batch_size=32)
# test_dl = DataLoader(test_data,batch_size=32)
#
# model = TransformerTimeSeries().to(device)
#
# lr = .0005 # learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# epochs = 100
#
# train_epoch_loss = []
# eval_epoch_loss = []
# Rp_best = 1e5
#
# torch.save(model.state_dict(), 'ConvTransformer_nologsparse_1.pth')
#
# # model_save_path = 'ConvTransformer_nologsparse.pth'
# for e, epoch in enumerate(range(epochs)):
#     train_loss = []
#     eval_loss = []
#
#     if (epoch % 10 is 0):
#
#         l_t = train_epoch(model, train_dl, t0=2073, future=231)
#         train_loss.append(l_t)
#
#         l_e = eval_epoch(model, val_dl, t0=1345, future=150)
#         eval_loss.append(l_e)
#
#         Rp = test_epoch(model, test_dl, t0=1345, future=150)
#
#         # if Rp_best > Rp:
#         #     Rp_best = Rp
#
#         train_epoch_loss.append(np.mean(train_loss))
#         eval_epoch_loss.append(np.mean(eval_loss))
#
#         print('-' * 89)
#         print("Epoch {}: Train loss: {} \t Validation loss: {} \t ".format(e,
#                                                                                 np.mean(train_loss),
#                                                                                  np.mean(eval_loss)))
#         print('-' * 89)
