import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.mem_transformer import MemTransformerLM
from model.gru_ae import *
import numpy as np
class Seq_Attention(BaseModel):
    def __init__(
        self,
        transformer_state_path,
        num_classes,
        codes=True,
        demographics=True,
        demographics_size=0,
        div_factor=2,
        dropout=0.5,
    ):
        super(Seq_Attention, self).__init__()

        self.num_classes = num_classes
        self.demographics = demographics
        self.demographics_size = demographics_size
        self.codes = codes

        state_dict = torch.load(transformer_state_path)
        transformer_config = state_dict["config"]
        state_dict = state_dict["state_dict"]

        transformer_args = transformer_config["model"]["args"]
        self.transformer = MemTransformerLM(**transformer_args)

        self.transformer.load_state_dict(state_dict)
        self.transformer.eval()

        self.patient_rep_size = +self.transformer.d_embed * int(
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
        #patient_rep = torch.Tensor([]).to(device)
        patient_rep = None
        with torch.no_grad():
            mem_out = self.transformer._forward(x_codes)
            mem_out = mem_out[x_cl, b_is, :]
        if self.codes and self.demographics:           
            patient_rep = torch.cat((mem_out, demo), dim=1)
        elif self.codes and not self.demographics:
            patient_rep = mem_out
        elif not self.codes and self.demographics:
            patient_rep = demo
        else:
            raise ValueError("codes and demographics can be false at the same time")
        # if self.demographics:
        #     if len(patient_rep.shape) == 0:
        #         patient_rep = demo
        #     else:
        #         if len(patient_rep.shape) == 1:
        #             patient_rep = patient_rep.unsqueeze(dim=0)
        #         patient_rep = torch.cat((patient_rep, demo), dim=1)

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