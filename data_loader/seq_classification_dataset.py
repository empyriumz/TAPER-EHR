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
        file_name = None,
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
        self.y_label = y_label
        self.data = pickle.load(open(os.path.join(data_path, file_name), "rb"))
        self.data_info = self.data["info"]
        self.data = self.data["data"]

        self.demographics_shape = self.data_info["demographics_shape"]

        self.keys = list(self.data.keys())
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
    
    def get_pos_weight(self):
        """The ratio of negative samples over positive samples

        Returns:
            [Float]: num_neg / num_pos
        """        
        pos_num = np.array([self.data[x][0][self.y_label] for x in self.train_idx]).sum()
        pos_weight = np.sqrt((len(self.train_idx) - pos_num) / pos_num)
        return pos_weight
        
    def _findmax_len(self):
        """Find the max number of visits of among all patients

        Returns:
            [int]: the max number of visits
        """        
        m = 0
        for v in self.data.values():
            if len(v) > m:
                m = len(v)
        return m-1

    def __getitem__(self, key):
        return self.preprocess(self.data[key])
    
    def __len__(self):
        return len(self.keys)
    
    def preprocess(self, seq):
        """n: total # of visits per each patients minus one
            it's also the index for the last visits for extracting label y[n]
        Args:
            key ([type]): [description]

        Returns:
            [type]: [description]
        """        
        n = len(seq) - 2
        x_codes = torch.zeros((self.num_codes, self.max_len), dtype=torch.float)
        demo = torch.Tensor(seq[0]["demographics"])
        for i, s in enumerate(seq[1:]):
            l = [
                    s["diagnoses"] * self.diag, 
                    s["procedures"] * self.proc
                ]
            
            codes = list(itertools.chain.from_iterable(l))
            x_codes[codes, i] = 1
            
            #codes, counts = np.unique(codes, return_counts=True)
            #x_codes[codes, i] = torch.tensor(counts, dtype=torch.float)                   

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
            y = torch.Tensor([seq[0]["readmission"]])
        else:
            y = torch.Tensor([seq[0]["mortality"]])

        return (x_codes.t(), x_cl, demo, y)


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