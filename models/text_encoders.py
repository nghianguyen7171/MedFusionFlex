"""
Advanced text encoders for clinical-biomedical NLP.

Add or replace encoders here and import FlexibleTextEncoder
from other modules (e.g., models.multimodal_model).
"""
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel,
    RobertaTokenizer, RobertaModel,
    DebertaV2Tokenizer, DebertaV2Model,
)

# ────────────────────────────────────────────────────────────
# 1. Domain-specific BERT variants
# ────────────────────────────────────────────────────────────
class _BaseBERTEncoder(nn.Module):
    def __init__(self, model_name: str, output_size: int = 256,
                 dropout: float = 0.2, freeze: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(self.backbone.config.hidden_size, output_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.drop(self.proj(cls))


class BioBERTEncoder(_BaseBERTEncoder):
    def __init__(self, output_size: int = 256):
        super().__init__('dmis-lab/biobert-base-cased-v1.1', output_size)


class BlueBERTEncoder(_BaseBERTEncoder):
    def __init__(self, output_size: int = 256):
        super().__init__(
            'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
            output_size,
            freeze=False  # fine-tune allowed
        )


class PubMedBERTEncoder(_BaseBERTEncoder):
    def __init__(self, output_size: int = 256):
        super().__init__(
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            output_size
        )

# ────────────────────────────────────────────────────────────
# 2. RoBERTa clinical variant
# ────────────────────────────────────────────────────────────
class ClinicalRoBERTaEncoder(nn.Module):
    def __init__(self, model_name: str = 'roberta-base',
                 output_size: int = 256, dropout: float = .2):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.backbone  = RobertaModel.from_pretrained(model_name)
        self.adapter   = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.ReLU(), nn.Dropout(.1),
            nn.Linear(512, self.backbone.config.hidden_size))
        self.proj  = nn.Linear(self.backbone.config.hidden_size, output_size)
        self.drop  = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        out  = self.backbone(input_ids, attention_mask).last_hidden_state
        cls  = out[:, 0, :] + self.adapter(out[:, 0, :])   # residual
        return self.drop(self.proj(cls))

# ────────────────────────────────────────────────────────────
# 3. DeBERTa medical variant
# ────────────────────────────────────────────────────────────
class MedicalDeBERTaEncoder(nn.Module):
    def __init__(self, model_name: str = 'microsoft/deberta-v3-base',
                 output_size: int = 256, dropout: float = .2):
        super().__init__()
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.backbone  = DebertaV2Model.from_pretrained(model_name)
        self.adapter   = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 1024),
            nn.GELU(), nn.Dropout(.1),
            nn.Linear(1024, self.backbone.config.hidden_size))
        self.gate  = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size,
                      self.backbone.config.hidden_size),
            nn.Sigmoid())
        self.proj = nn.Linear(self.backbone.config.hidden_size, output_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids, attention_mask).last_hidden_state
        adapt = self.adapter(out)
        gated = self.gate(out) * adapt + (1 - self.gate(out)) * out
        cls   = gated[:, 0, :]
        return self.drop(self.proj(cls))

# ────────────────────────────────────────────────────────────
# 4. Flexible factory
# ────────────────────────────────────────────────────────────
_ENCODER_MAP = {
    'clinicalbert' : lambda sz: _BaseBERTEncoder(
        'emilyalsentzer/Bio_ClinicalBERT', sz),
    'biobert'      : BioBERTEncoder,
    'bluebert'     : BlueBERTEncoder,
    'pubmedbert'   : PubMedBERTEncoder,
    'roberta_clin' : ClinicalRoBERTaEncoder,
    'deberta_med'  : MedicalDeBERTaEncoder,
}

class FlexibleTextEncoder(nn.Module):
    """
    Drop-in replacement for any text branch.  Usage:
        enc = FlexibleTextEncoder('biobert', output_size=256)
        z   = enc(input_ids, attention_mask)
    """
    def __init__(self, encoder_type: str = 'clinicalbert', output_size: int = 256):
        super().__init__()
        if encoder_type not in _ENCODER_MAP:
            raise ValueError(f'Unsupported encoder {encoder_type}')
        self.inner = _ENCODER_MAP[encoder_type](output_size)

    @property
    def tokenizer(self):
        return self.inner.tokenizer

    def forward(self, input_ids, attention_mask):
        return self.inner(input_ids, attention_mask)
