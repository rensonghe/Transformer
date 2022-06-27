#%%
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

class time_series_decoder_paper(Dataset):
    """synthetic time series dataset from section 5.1"""

    def __init__(self, source_data, transform=None):
        """
        Args:
            t0: previous t0 data points to predict from
            N: number of data points
            transform: any transformations to be applied to time series
        """
        self.future = int(source_data.shape[0]*0.001+0.5)  #15个点，每个点2min，预测未来30分钟
        self.t0 = int(source_data.shape[0]*0.999+0.5)   #未来两年的数据预测未来
        self.N =source_data.shape[1]  #一共有多少列 时间序列拿来训练
        self.transform = None

        # define time points
        self.x = torch.cat(self.N * [torch.arange(0,self.t0 + self.future).type(torch.float).unsqueeze(0)])  #(N, 多少行数据)
        self.data = torch.tensor(np.array(source_data.T.reset_index(drop=True)),dtype=torch.float)
        # add noise
        self.data = self.data + torch.randn(self.data.shape)

        self.masks = self._generate_square_subsequent_mask()

        # print out shapes to confirm desired output
        print("x: {}*{}".format(*list(self.x.shape)),
              "data: {}*{}".format(*list(self.data.shape)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.x[idx, :],
                  self.data[idx, :],
                  self.masks)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _generate_square_subsequent_mask(self):
        mask = torch.zeros(self.t0 + self.future, self.t0 + self.future)

        for i in range(0, self.t0):
            mask[i, self.t0:] = 1
        for i in range(self.t0, self.t0 + self.future):
            mask[i, i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))  # .masked_fill(mask == 1, float(0.0))
        # print(mask.shape,'mask_shape')
        return mask