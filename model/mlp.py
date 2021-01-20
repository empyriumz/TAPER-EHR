import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.gru_ae import *
import numpy as np

class Seq_MLP(BaseModel):
    def __init__(
        self,
        num_classes,
        codes=True,
        demographics=True,
        num_codes=0,
        num_visits=0,
        demographics_size=0,
        div_factor=2,
        dropout=0.5,
    ):
        super(Seq_MLP, self).__init__()

        self.num_classes = num_classes
        self.demographics = demographics
        self.demographics_size = demographics_size
        self.codes = codes
        self.num_codes = num_codes
        self.num_visits = num_visits
        self.patient_rep_size = + self.num_visits * self.num_codes * int(
            self.codes
        ) + self.demographics_size * int(self.demographics)
        self.predictor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                self.patient_rep_size,
                self.patient_rep_size // div_factor,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(self.patient_rep_size // div_factor, self.num_classes),
        )

    def forward(self, x, device="cuda"):
        x_codes, x_cl, b_is, demo = x

        x_codes = x_codes.to(device)
        x_cl = x_cl.to(device)
        demo = demo.to(device)
        b_is = b_is.to(device)
        patient_rep = None
        batch_size = x_codes.shape[1]
        if self.codes:
            patient_rep = x_codes.transpose(0, 1).reshape(batch_size, -1)

        if self.demographics:
            if len(patient_rep.shape) == 0:
                patient_rep = demo
            else:
                if len(patient_rep.shape) == 1:
                    patient_rep = patient_rep.unsqueeze(dim=0)
                patient_rep = torch.cat((patient_rep, demo), dim=1)

        logits = self.predictor(patient_rep)
        if self.num_classes > 1:
            log_probs = F.log_softmax(logits, dim=1).squeeze()
        else:            
            log_probs = torch.sigmoid(logits)
            
        if len(logits) == 1:
            logits = logits.squeeze(dim=0)
            log_probs = log_probs.squeeze(dim=0)
        else:
            logits = logits.squeeze()
            log_probs = log_probs.squeeze()
                     
        return log_probs, logits

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters if p is not None])
        return "\nTrainable parameters: {}".format(params)