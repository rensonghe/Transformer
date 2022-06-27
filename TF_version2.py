#%%
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


torch.manual_seed(0)
np.random.seed(0)

# This concept is also called teacher forceing.
# The flag decides if the loss will be calculted over all
# or just the predicted values.
calculate_loss_over_all_values = False
#%%
# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)
#
# print(out)
#%%
input_window = 120
output_window = 30
batch_size = 128  # batch size
device = torch.device("cuda")

#%%
d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 1    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8
#
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table)              # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):                                         # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)

def get_attn_pad_mask(seq_q, seq_k):                       # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q ,_= seq_q.size()         #1*64*5
    batch_size, len_k ,_= seq_k.size()
    # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)          # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    pad_attn_mask = torch.ones(batch_size,len_q,len_k)     # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    return pad_attn_mask  # 扩展成多维度

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):  # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):  # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)  # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model)(output + residual)  # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                                     # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                                        # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):                                # enc_inputs: [batch_size, src_len, d_model]
        #输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V                          # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,    # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)                    # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)                                       # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Linear(5, d_model)
        self.pos_emb = PositionalEncoding(d_model)                               # 加入位置信息
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):                                               # enc_inputs: [batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs)   # enc_outputs: [batch_size, src_len, d_model]
        # print('1',enc_outputs)
        enc_outputs = self.pos_emb(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],                                                                # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder()
        self.projection = nn.Linear(d_model, 1, bias=False)
        # self.projection1 = nn.Linear(128, 1, bias=False)
    def forward(self, enc_inputs):                         # enc_inputs: [batch_size, src_len]                                                   # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)         # enc_outputs: [batch_size, src_len, d_model],
                                                                       # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_logits = self.projection(enc_outputs)                      # dec_logits: [batch_size, tgt_len, 1]
        # dec_logits = self.projection1(dec_logits)                      # dec_logits: [batch_size, tgt_len, 1]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns


#%%
data_path = r'test.csv'
#%%
elements=[]

#%%
def single_data():

    data_all = pd.read_csv('')
    data_ha = []
    length = len(data_all)
    for index, element in enumerate(elements):
        data_element = data_all[element].values.astype(np.float64)
        data_element = data_element.reshape(length,1)
        data_ha.append(data_element)
    X_hat = np.concatenate(data_ha,axis=1)
    max1 = np.max(X_hat, axis=0)
    X_CONVERT = torch.from_numpy(X_hat)
    X = torch.zeros_like(X_CONVERT)
    a = len(X_CONVERT)
    for i in range(a):
        X[i, : ] = X_CONVERT[a - 1 - i, : ]
    y = X[5:,3].type(torch.float32)
    y = y.reshape(y.shape[0],1)
    X = X[0:-5,:].type(torch.float32)
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return data_loader

def data_load(seq_len):
    x, y = single_data()
    len = x.shape[0]
    data_last_index = len - seq_len
    X = []
    Y = []
    for i in range(0, data_last_index, 5):
        data_x = np.expand_dims(x[i:i + seq_len], 0)  # [1,seq,feature_size]
        data_y = np.expand_dims(y[i:i + seq_len], 0)  # [1,seq,out_size]
        # data_y=np.expand_dims(y[,0)   #[1,seq,out_size]
        X.append(data_x)
        Y.append(data_y)
    data_x = np.concatenate(X, axis=0)
    data_y = np.concatenate(Y, axis=0)
    data = torch.from_numpy(data_x).type(torch.float32)
    label = torch.from_numpy(data_y).type(torch.float32)
    return data, label  # [num_data,seq,feature_size]  [num_data,seq] 默认out_size为1

def dataset(seq_len, batch):
    X, Y = data_load(seq_len)
    feature_size = X.shape[-1]
    out_size = Y.shape[-1]
    dataset_train = TensorDataset(X, Y)
    dataloader = DataLoader(dataset_train, batch_size=batch, shuffle=False)
    return dataloader, feature_size, out_size










#%%
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
#%%
# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    # print(L)
    # np.zeros((output_window,3))
    for i in range(L-tw):
        train_seq = np.append(input_data[i:i+tw,:][:-output_window,:] , np.zeros((output_window,10)),axis=0)
        train_label = input_data[i:i+tw,:]
        # print(train_seq.shape,train_label.shape)
        #train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)
#%%
data = pd.read_csv('test.csv')
# data = data.drop(['buyratio','sellratio','tradingprice','tradingvolume'],axis=1)
#%%
# data['target'] = np.log(data['close'] / data['close'].shift(1))
# data['target'][np.isinf(data['target'])] = 0
# data = data.dropna(axis=0, how='any')
data = data.iloc[:,26:]
#%%
def get_data():
    # time        = np.arange(0, 400, 0.1)
    # amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))

    # from pandas import read_csv
    # series = read_csv('data.csv', header=0, index_col='time', parse_dates=True,
    #                   squeeze=True)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    series = data.to_numpy()

    amplitude = scaler.fit_transform(series)
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    # sampels = 80000
    train_data = amplitude[:100000]
    test_data = amplitude[-4000:]
    print(len(train_data))
    print(len(test_data))

    train_sequence = create_inout_sequences(train_data, input_window)
    # print('a',train_sequence.size())
    train_sequence = train_sequence[:-output_window]  # todo: fix hack?

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]  # todo: fix hack?

    return train_sequence.to(device), test_data.to(device), scaler

#%%
def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1)).squeeze()  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1)).squeeze()
    return input, target
#%%
train_data, val_data,scaler = get_data()
print(train_data.size())
# print(train_data.size(), val_data.size())
tr,te = get_batch(train_data, 0,batch_size)
print(tr.shape,te.shape)

#%%
def train(train_data):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
#%%
def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)
            print(output[-output_window:].size(),targets[-output_window:].size())
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)
#%%

def plot(eval_model, data_source, epoch, scaler):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)
            # look like the model returns static values for the output window
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1, :].squeeze(1).cpu()),
                                    0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1, :].squeeze(1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    len(test_result)

    test_result_ = scaler.inverse_transform(test_result[:700])
    truth_ = scaler.inverse_transform(truth)
    print(test_result.shape, truth.shape)
    for m in range(9):
        test_result = test_result_[:, m]
        truth = truth_[:, m]
        fig = pyplot.figure(1, figsize=(20, 5))
        fig.patch.set_facecolor('xkcd:white')
        pyplot.plot([k + 510 for k in range(190)], test_result[510:], color="red")
        pyplot.title('Prediction uncertainty')
        pyplot.plot(truth[:700], color="black")
        pyplot.legend(["prediction", "true"], loc="upper left")
        ymin, ymax = pyplot.ylim()
        pyplot.vlines(510, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        pyplot.ylim(ymin, ymax)
        pyplot.xlabel("Periods")
        pyplot.ylabel("Y")
        pyplot.show()
        pyplot.close()
    return total_loss / i
#%%
model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.005
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = float("inf")
epochs = 50  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    if (epoch % 1 is 0):
        val_loss = plot(model, val_data, epoch, scaler)
        # predict_future(model, val_data,200,epoch,scaler)
    else:
        val_loss = evaluate(model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                time.time() - epoch_start_time),
                                                                                                  val_loss,
                                                                                                  math.exp(val_loss)))
    print('-' * 89)

    # if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step()

#%%
a = torch.exp(torch.arange(0, 10, 2).float() * (-math.log(10000.0) / 10))
#%%
b = torch.exp(torch.arange(0, 35, 2).float() * (-math.log(10000.0) / 35))