import numpy as np
import torch
import matplotlib.pyplot as plt
import causal_convolution_layer
import Dataloader
import math
from torch.utils.data import DataLoader

#%%
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
        self.input_embedding = causal_convolution_layer.context_embedding(2, 256, 9)
        self.positional_embedding = torch.nn.Embedding(512, 256)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)

        self.fc1 = torch.nn.Linear(256, 1)

    def forward(self, x, y, attention_masks):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        z = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)
        print(x.shape, 'xxxx')
        print(y.shape, 'yyyy')

        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)

        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1, 0, 2)

        input_embedding = z_embedding + positional_embeddings

        transformer_embedding = self.transformer_decoder(input_embedding, attention_masks)

        output = self.fc1(transformer_embedding.permute(1, 0, 2))

        return output

#%%
t0 = 24
# train_dataset = Dataloader.time_series_decoder_missing_paper(t0,4500) # missing
# validation_dataset = Dataloader.time_series_decoder_missing_paper(t0,500) # missing
# test_dataset = Dataloader.time_series_decoder_missing_paper(t0,1000) # missing

train_dataset = Dataloader.time_series_decoder_paper(t0, 4500)
validation_dataset = Dataloader.time_series_decoder_paper(t0, 500)
test_dataset = Dataloader.time_series_decoder_paper(t0, 1000)
#%%
criterion = torch.nn.MSELoss()
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dl = DataLoader(validation_dataset, batch_size=64)
test_dl = DataLoader(test_dataset, batch_size=128)

# lr = .0005  # learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# epochs = 100

#%%
def Dp(y_pred, y_true, q):
    return max([q * (y_pred - y_true), (q - 1) * (y_pred - y_true)])


def Rp_num_den(y_preds, y_trues, q):
    numerator = np.sum([Dp(y_pred, y_true, q) for y_pred, y_true in zip(y_preds, y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator, denominator
#%%
# from parallel import BalancedDataParallel
# batch_szie = 32
# gpu0_bsz = 8
# acc_grad = 2
model = TransformerTimeSeries().cuda()
# model = BalancedDataParallel(gpu0_bsz // acc_grad, model, dim=1).cuda()
lr = .0005  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 10
#%%
# def train_epoch(model, train_dl, t0=96):
model.train()
train_loss = 0
n = 0
for step, (x, y, attention_masks) in enumerate(train_dl):
    optimizer.zero_grad()
    output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())
    loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + 24 - 1)], y.cuda()[:, t0:])  # not missing data
    # loss = criterion(output.squeeze()[:,(t0-1-10):(t0+24-1-10)],y.cuda()[:,(t0-10):]) # missing data
    loss.backward()
    optimizer.step()

    train_loss += (loss.detach().cpu().item() * x.shape[0])
    n += x.shape[0]



#%%
# train_epoch_loss = []
# eval_epoch_loss = []
# Rp_best = 10
# for e, epoch in enumerate(range(epochs)):
#     train_loss = []
#     eval_loss = []
#
#     l_t = train_epoch(model, train_dl, t0)
#     train_loss.append(l_t)
#
#     # l_e = eval_epoch(model, validation_dl, t0)
#     # eval_loss.append(l_e)
#     #
#     # Rp = test_epoch(model, test_dl, t0)
#     #
#     # if Rp_best > Rp:
#     #     Rp_best = Rp
#     #
#     # train_epoch_loss.append(np.mean(train_loss))
#     # eval_epoch_loss.append(np.mean(eval_loss))
#     #
#     # print("Epoch {}: Train loss: {} \t Validation loss: {} \t R_p={}".format(e,
#     #                                                                          np.mean(train_loss),
#     #                                                                          np.mean(eval_loss), Rp))