import torch
import torch.utils.data as data
import os
import pickle
import itertools

class SeqCodeDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        batch_size,
        train=True,
        med=False,
        cpt=False,
        diag=False,
        proc=False,
        split_num=2,
    ):
        self.proc = proc
        self.med = med
        self.diag = diag
        self.cpt = cpt 
        self.train = train
        self.batch_size = batch_size

        self.data = pickle.load(open(os.path.join(data_path, "data_icd.pkl"), "rb"))
        self.data_info = self.data["info"]
        self.data = self.data["data"]

        data_split_path = os.path.join(
            data_path, "splits", "split_{}.pkl".format(split_num)
        )
        if os.path.exists(data_split_path):
            self.train_idx, self.valid_idx = pickle.load(open(data_split_path, "rb"))

        self.keys = self._get_keys()

        self.max_len = self._findmax_len()

        self.num_dcodes = self.data_info['num_icd9_codes']
        self.num_pcodes = self.data_info['num_proc_codes']
    
        self.num_codes = (
            self.diag * self.num_dcodes
            + self.proc * self.num_pcodes
        )

        self.demographics_shape = self.data_info["demographics_shape"]

    def _gen_idx(self, keys, min_adm=2):
        idx = []
        for k in keys:
            v = self.data[k]
            if len(v) < min_adm:
                continue
            for i, _ in enumerate(v):
                idx.append((k, i))
        return idx

    def _get_keys(self, min_adm=2):
        keys = []
        for k, v in self.data.items():
            if len(v) < min_adm:
                continue
            keys.append(k)
        return keys

    def _findmax_len(self):
        m = 0
        for v in self.data.values():
            if len(v) > m:
                m = len(v)
        return m

    def __len__(self):
        if self.train:
            return len(self.keys)
        else:
            return 0

    def __getitem__(self, k):
        x = self.preprocess(self.data[k])
        return x

    
    def preprocess(self, seq):
        """create one hot vector of idx in seq, with length self.num_codes

        Args:
            seq: list of ideces where code should be 1

        Returns:
            x: one hot vector
            ivec: vector for learning code representation
            jvec: vector for learning code representation
        """

        icd_one_hot = torch.zeros((self.num_codes, self.max_len), dtype=torch.long)
        demo_one_hot = torch.zeros((self.demographics_shape, self.max_len), dtype=torch.long)
        mask = torch.zeros((self.max_len,), dtype=torch.long)
        ivec = []
        jvec = []
        for i, s in enumerate(seq):
            demo = s["demographics"]
            l = [
                 s["diagnoses"] * self.diag, 
                 s["procedures"] * self.proc
            ]
            icd = list(itertools.chain.from_iterable(l))
            
            icd_one_hot[icd, i] = 1
            demo_one_hot[:, i] = torch.Tensor(demo)
            mask[i] = 1
            for j in icd:
                for k in icd:
                    if j == k:
                        continue
                    ivec.append(j)
                    jvec.append(k)
        return icd_one_hot.t(), mask, torch.LongTensor(ivec), torch.LongTensor(jvec), demo_one_hot.t()

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
    x, m, ivec, jvec, demo = zip(*data)
    m = torch.stack(m, dim=1)
    x = torch.stack(x, dim=1) 
    ivec = torch.cat(ivec, dim=0)
    jvec = torch.cat(jvec, dim=0)
    demo = torch.stack(demo, dim=1)
    return x, m, ivec, jvec, demo
