"""Ordinary CLS/pooler binary classification; all BERT weights trainable."""
import hashlib
import random
import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertConfig


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)


class Classifier(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size,2)
        nn.init.normal_(self.classifier.weight, std=bert.config.initializer_range)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, inputs):
        ids, mask, segments = inputs
        positions = torch.arange(ids.shape[1],device=ids.device).expand_as(ids)
        output = self.bert(input_ids=ids,attention_mask=mask,token_type_ids=segments,
                           position_ids=positions,return_dict=True)
        return self.classifier(output.pooler_output)


def make_model(model_path, dropout=.1):
    cfg = BertConfig.from_pretrained(str(model_path),local_files_only=True)
    cfg.hidden_dropout_prob = cfg.attention_probs_dropout_prob = dropout
    cfg._attn_implementation = 'eager'
    return Classifier(BertModel.from_pretrained(str(model_path),config=cfg,local_files_only=True)).train()


def digest(model):
    h = hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
