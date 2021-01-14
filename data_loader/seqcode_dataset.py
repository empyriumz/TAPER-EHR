import torch
import torch.utils.data as data
import os
import pickle
import itertools
import numpy as np
class SeqCodeDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        batch_size,
        file_name = None,
        med=False,
        cpt=False,
        diag=False,
        proc=False
    ):
        self.proc = proc
        self.med = med
        self.diag = diag
        self.cpt = cpt 
        self.batch_size = batch_size
        self.data = pickle.load(open(os.path.join(data_path, file_name), "rb"))
        self.data_info = self.data["info"]
        self.data = self.data["data"]

        self.keys = list(self.data.keys())
        self.max_len = self._findmax_len()
        self.num_dcodes = self.data_info['num_icd9_codes']
        self.num_pcodes = self.data_info['num_proc_codes']
    
        self.num_codes = (
            self.diag * self.num_dcodes
            + self.proc * self.num_pcodes
        )

    def _findmax_len(self):
        m = 0
        for v in self.data.values():
            if len(v) > m:
                m = len(v)
        # substract one which contains demo info instead of visits 
        return m-1

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, k):
        return self.preprocess(self.data[k])

    def preprocess(self, seq):
        """create one hot vector of idx in seq, with length self.num_codes

        Args:
            seq: list of ideces where code should be 1

        Returns:
            x: one hot vector
            ivec: vector for learning code representation
            jvec: vector for learning code representation
        """

        codes_one_hot = torch.zeros((self.num_codes, self.max_len), dtype=torch.long)
        mask = torch.zeros((self.max_len,), dtype=torch.long)
        ivec = []
        jvec = []
        for i, s in enumerate(seq[1:]):
            l = [
                    s["diagnoses"] * self.diag, 
                    s["procedures"] * self.proc
                ]
            codes = list(set(itertools.chain.from_iterable(l)))
            codes_one_hot[codes, i] = 1
            
            # codes, counts = np.unique(codes, return_counts=True)
            # codes_one_hot[codes, i] = torch.LongTensor(counts)
            
            mask[i] = 1
            for j in codes:
                for k in codes:
                    if j == k:
                        continue
                    ivec.append(j)
                    jvec.append(k)
        return codes_one_hot.t(), mask, torch.LongTensor(ivec), torch.LongTensor(jvec)

def collate_fn(data):
    """Creates mini-batch from x, ivec, jvec tensors

    We should build custom collate_fn, as the ivec, and jvec have varying lengths. These should be appended
    in row form

    Args:
        data: list of tuples contianing (x, ivec, jvec)

    Returns:
        x: one hot encoded vectors stacked vertically
        ivec: long vector
        jvec: long vector
    """
    x, m, ivec, jvec = zip(*data)
    m = torch.stack(m, dim=1)
    x = torch.stack(x, dim=1) 
    ivec = torch.cat(ivec, dim=0)
    jvec = torch.cat(jvec, dim=0)
    return x, m, ivec, jvec
