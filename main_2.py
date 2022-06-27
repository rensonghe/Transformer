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
print(torch.cuda.is_available())
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
        self.input_embedding = causal_convolution_layer.context_embedding(2, 256, 6)  #定义conv1函数的是图像卷积函数：输入为2个频道，,输出为256张特征图, 卷积核为9*9 正方形
        self.positional_embedding = torch.nn.Embedding(15000,256)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=2)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=2)

        self.fc1 = torch.nn.Linear(256, 1)
        self.drop_path_prob = 0.0


    def forward(self, x, y, attention_masks):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        z = torch.cat((y.unsqueeze(1),x.unsqueeze(1)),1)
        # print(z.shape,'zzzzzzzz')
        # print(y.unsqueeze(1).shape, 'yyyyyyyyyy')
        # print(x.unsqueeze(1).shape, 'xxxxxxxxx')
        # print(attention_masks.shape,'sssssssss')

        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)


        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1, 0, 2)

        input_embedding = z_embedding + positional_embeddings

        transformer_embedding = self.transformer_decoder(input_embedding, attention_masks)
        output = self.fc1(transformer_embedding.permute(1, 0, 2))
        # print(output.shape, 'shape')

        return output

#%%
# 数据加载
data = pd.read_csv('test.csv')
# data = data.drop(['buyratio','sellratio','tradingprice','tradingvolume'],axis=1)
#%%
data['target'] = np.log(data['close'] / data['close'].shift(1))
data['target'][np.isinf(data['target'])] = 0
data = data.dropna(axis=0, how='any')
data = data.iloc[:,1:]

#%%

scaler= StandardScaler().fit(data)
# 保存归一化模型
pickle.dump(scaler, open('scale_model.pth', 'wb'))
print('Scaler model saved to {}'.format('scaler_model.pth'))

#%%
# train_data = data[(data.index>='2019-07-01 0:00:00')& (data.index<='2020-04-01 0:00:00')]
# test_data = data[(data.index>='2021-10-01 0:00:00')& (data.index<='2021-11-10 23:59:59')]
train_data = data[:6000]
test_data = data[-4000:]

#%%
# t_time = train_data.index
# close = train_data['sellvolume']
# plt.figure(figsize=(20,8), dpi=72)
# plt.plot(t_time,close,label='sellvolume')
# plt.legend(loc=0, frameon=True)
# plt.ylabel('sellvolume')
# # plt.savefig('exchange.png')
# plt.show()
#%%
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
from torch import nn
# device = torch.device('cuda')
model = TransformerTimeSeries()
#%% 单GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = TransformerTimeSeries().cuda()

#%% 多GPU
model = nn.DataParallel(model)
model = model.cuda()
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
     print("Let's use", torch.cuda.device_count(), "GPUs!")


#%%
train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=128)

#%%
# t0=4901
# future=5
# def train_epoch(model, train_dl, t0, future):
# model.train()
# train_loss = 0
# n = 0
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
#%%

# for step, (x, y, attention_masks) in enumerate(train_dl):
#     optimizer.zero_grad()
#
#     # model = DataParallelModel(model)
#     # parallel_loss = DataParallelCriterion(criterion)
#     print(x.cuda())
#     print(y.cuda())
#     print(x.shape)
#     print(y.shape)
#     attention = attention_masks[0]
#     attention = torch.cat((attention,attention,attention))
#     print(attention.shape)
#     output = model(x.cuda(), y.cuda(), attention.cuda())
#     loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + future - 1)], y[:, t0:].cuda())  # not missing data
#     # loss = criterion(output.squeeze()[:,(t0-1-10):(t0+24-1-10)],y.cuda()[:,(t0-10):]) # missing data
#     loss.backward()
#     optimizer.step()

    # train_loss += (loss.detach().cpu().item() * x.shape[0])
    # n += x.shape[0]

#%%
def train_epoch(model, train_dl, t0, future):
    model.train()
    train_loss = 0
    n = 0
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for step, (x, y, attention_masks) in enumerate(train_dl):

        optimizer.zero_grad()
        attention = attention_masks[0]
        attention = torch.cat((attention,attention))

        output = model(x.cuda(), y.cuda(), attention.cuda())
        loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + future - 1)], y[:, t0:].cuda())  # not missing data
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

            attention = attention_masks[0]
            attention = torch.cat((attention,attention))

            output = model(x.cuda(), y.cuda(), attention.cuda())

            for p, o in zip(output.squeeze()[:, (t0 - 1):(t0 + future - 1)].cpu().numpy().tolist(),
                            y[:, t0:].cpu().numpy().tolist()):  # not missing data
                # for p,o in zip(output.squeeze()[:,(t0-1-10):(t0+24-1-10)].cpu().numpy().tolist(),y.cuda()[:,(t0-10):].cpu().numpy().tolist()): # missing data

                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            # plot_result(hist, y_preds, y_trues, t0)
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den

    return Rp

#%%
train_epoch_loss = []
test_epoch_loss = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
Rp_best = 1e5
epochs = 50
model_save_path = 'ConvTransformer_nologsparse.pth'
for e, epoch in enumerate(range(epochs)):
    train_loss = []
    test_loss = []

    l_t = train_epoch(model, train_dl, t0=5994, future=6)
    train_loss.append(l_t)

    Rp = test_epoch(model, test_dl, t0=3996, future=4)

    if Rp_best > Rp:
        Rp_best = Rp
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': Rp,
        }, model_save_path)

    train_epoch_loss.append(np.mean(train_loss))
    test_epoch_loss.append(np.mean(test_loss))

    print("Epoch {}: Train loss: {} \t Validation loss: {} \t R_p={}".format(e,
                                                                             np.mean(train_loss),
                                                                             np.mean(test_loss), Rp))

    print("Rp best={}".format(Rp_best))
    torch.save(model, model_save_path)



#%%
def plot_result(history, yhat, ytruth, t0):
    # 带上历史值
    yhat = history + yhat
    ytruth = history + ytruth
    # 画图
    x = range(len(ytruth))
    yhat = np.round(yhat, 2)
    ytruth = np.round(ytruth, 2)
    plt.figure(facecolor='w')
    plt.plot(range(len(x)), ytruth, 'green', linewidth=1.5, label='ground truth')
    plt.plot(range(len(x)), yhat, 'blue', alpha=0.8, linewidth=1.2, label='predict value')
    # 画条预测起始线
    plt.vlines(t0, yhat.min() * 0.99, yhat.max() * 1.01,
               alpha=0.7, colors="r", linestyles="dashed")
    # plt.text(0.15, 0.01, error_message, size=10, alpha=0.9, transform=plt.gca().transAxes)  # 相对位置，经验设置值
    plt.legend(loc='best')  # 设置标签的位置
    plt.grid(True)
    plt.show()
#%%
def prediction(model, dl, t0, future):
    # 预测前先load model， dl就是待预测数据，t0就是前n和时间点，future就是要预测的n个时间点
    # 比如你要用一周内前五天的数据训练模型，来预测后两天的值 t0 = 5 * 24 = 120， future = 48
    with torch.no_grad():
        predictions = []
        observations = []
        for step, (x, y, attention_masks) in enumerate(dl):
            # x: (batch_size， total_ts_length)
            # y: (batch_size, total_ts_length)
            # ouput:(batch_size, total_ts_length, 1)
            model = torch.load(model_save_path)
            attention = attention_masks[0]
            attention = torch.cat((attention,attention))
            output = model(x.cuda(), y.cuda(), attention.cuda())
            history = y[:, :t0].cpu().numpy().tolist()
            for p, o in zip(output.squeeze()[:, (t0 - 1):(t0 + future - 1)].cpu().numpy().tolist(),
                            y[:, t0:].cpu().numpy().tolist()):  # not missing data

                predictions.append(p) # (batch_size, future)
                observations.append(o) # (batch_size, future)
                # print(predictions)
                print(observations)
        num = 0
        den = 0
        for hist, y_preds, y_trues in zip(history, predictions, observations):
            # print(y_preds.shape,'shape')
            # print(len(y_preds))
            plot_result(hist, y_preds, y_trues, t0)
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den
    return Rp

#%%
model_test = torch.load('ConvTransformer_nologsparse.pth')
#%%
prediction(model_test, test_dl, t0=3000, future=1000)