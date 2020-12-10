import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import itertools

class SeqClassificationDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        batch_size,
        y_label="los",
        train=True,
        balanced_data=False,
        validation_split=0.0,
        split_num=1,
        med=False,
        diag=True,
        proc=True,
        cptcode=False
    ):
        super(SeqClassificationDataset).__init__()
        self.proc = proc
        self.med = med
        self.diag = diag
        self.cpt = cptcode

        self.data_path = data_path
        self.batch_size = batch_size
        self.train = train
        self.y_label = y_label
        self.validation_split = validation_split
        self.balanced_data = balanced_data
        self.data = pickle.load(open(os.path.join(self.data_path, "data_icd.pkl"), "rb"))
        self.data_info = self.data["info"]
        self.data = self.data["data"]

        self.demographics_shape = self.data_info["demographics_shape"]

        self.keys = list(map(int, self.data.keys()))
        self.max_len = self._findmax_len()

        self.num_dcodes = self.data_info['num_icd9_codes']
        self.num_pcodes = self.data_info['num_proc_codes']
        self.num_mcodes = self.data_info['num_med_codes']
        self.num_ccodes = self.data_info['num_cpt_codes']
        
        self.num_codes = (
            self.diag * self.num_dcodes
            + self.cpt * self.num_ccodes
            + self.proc * self.num_pcodes
            + self.med * self.num_mcodes
        )  

        data_split_path = os.path.join(
            self.data_path, "splits", "split_{}.pkl".format(split_num)
        )
        if os.path.exists(data_split_path):
            self.train_idx, self.valid_idx = pickle.load(open(data_split_path, "rb"))
            # select patients with at least two admissions
            self.train_indices = self._gen_indices(self.train_idx)
            self.valid_indices = self._gen_indices(self.valid_idx)
            # re-label the patient ID!
            # only patients with at least two visits are kept
            self.train_idx = np.arange(len(self.train_indices))
            self.valid_idx = len(self.train_indices) + np.arange(len(self.valid_indices))

            if self.balanced_data:
                self.train_idx = self._gen_balanced_indices(self.train_idx)
                #self.valid_idx = self._gen_balanced_indices(self.valid_idx)
        else:
            # TODO: data index logic if train, validation splits are not provided
            pass

    def _gen_balanced_indices(self, indices):
        """Generate a balanced set of indices"""
        ind_idx = {}

        for idx in indices:
            label = self.get_label(idx)
            if label not in ind_idx:
                ind_idx[label] = [idx]
            else:
                ind_idx[label].append(idx)

        tr = []
        te = []

        lens = sorted([len(v) for v in ind_idx.values()])

        if len(lens) > 3:
            num_samples = lens[-2]
        else:
            num_samples = lens[0]

        for v in ind_idx.values():
            v = np.asarray(v)

            if len(v) > num_samples:
                v = v[np.random.choice(np.arange(len(v)), num_samples)]

            # train, test = train_test_split(v, test_size=self.validation_split, random_state=1)
            # te.append(test)

            tr.append(v)

        train = np.concatenate(tr)
        # test = np.concatenate(te)
        return train  # , test

    def _gen_indices(self, keys):
        indices = []
        for k in keys:
            v = self.data[k]
            for j in range(len(v)):
                if (j + 1) == len(v):
                    continue
                indices.append([k, j + 1])
        return indices
    
    def _findmax_len(self):
        """Find the max number of visits of any patients

        Returns:
            [int]: the max number of visits
        """        
        m = 0
        for v in self.data.values():
            if len(v) > m:
                m = len(v)
        return m

    def __getitem__(self, index):
        if index in self.train_idx:
            idx = self.train_indices[index]
        else:
            idx = self.valid_indices[index - len(self.train_indices)]
        x = self.preprocess(idx)
        return x

    def preprocess(self, idx):
        """n: total # of visits per each patients minus one
            it's also the index for the last visits for extracting label y[n]
        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """        
        seq = self.data[idx[0]]
        n = idx[1]
        x_codes = torch.zeros((self.num_codes, self.max_len), dtype=torch.float)
        demo = torch.Tensor(seq[n]["demographics"])
        for i in range(n):
            if (i + 1) == len(seq):
                continue
            s = seq[i]   
            codes = [
                 s["diagnoses"] * self.diag, 
                 s["procedures"] * self.proc
            ]
            codes = list(itertools.chain.from_iterable(codes))
            x_codes[codes, i] = 1

        x_cl = torch.Tensor(
            [
                n,
            ]
        )
       
        if self.y_label == "los":
            los = seq[n]["los"]
            if los != los:
                los = 9
            y = torch.Tensor([los - 1])
        elif self.y_label == "readmission":
            y = torch.Tensor([seq[n]["readmission"]])
        else:
            y = torch.Tensor([seq[n]["mortality"]])

        return (x_codes.t(), x_cl, demo, y)

    def get_label(self, idx):
        if idx in self.train_idx:
            idx = self.train_indices[idx]
        else:
            idx = self.valid_indices[idx - len(self.train_indices)]
        seq = self.data[idx[0]]
        n = idx[1]
        if self.y_label == "los":
            los = seq[n]["los"]
            if los != los:
                los = 9
            y = torch.Tensor([los - 1])
        elif self.y_label == "readmission":
            y = torch.Tensor([seq[n]["readmission"]])
        else:
            y = torch.Tensor([seq[n]["mortality"]])
        y = y.item()
        return y

    def __len__(self):
        l = 0
        if self.train:
            l = len(self.train_idx)
        else:
            l = len(self.valid_idx)

        return l

def collate_fn(data):
    x_codes, x_cl,  demo, y_code = zip(*data)
    x_codes = torch.stack(x_codes, dim=1)
    demo = torch.stack(demo, dim=0)
    y_code = torch.stack(y_code, dim=1).long()
    x_cl = torch.stack(x_cl, dim=0).long()
    b_is = torch.arange(x_cl.shape[0]).reshape(tuple(x_cl.shape)).long()
    return (
        x_codes,
        x_cl.squeeze(),
        b_is.squeeze(),
        demo,
    ), y_code.squeeze()