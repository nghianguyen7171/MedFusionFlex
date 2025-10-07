"""
Per-encoder hyper-parameters.
"""
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class EncoderConfig:
    model_name: str
    max_length: int = 512
    output_size: int = 256
    freeze_params: bool = True
    dropout: float = 0.2
    notes: Optional[str] = None

ENCODER_CONFIGS: Dict[str, EncoderConfig] = {
    'clinicalbert': EncoderConfig(
        'emilyalsentzer/Bio_ClinicalBERT', freeze_params=True),
    'biobert': EncoderConfig(
        'dmis-lab/biobert-base-cased-v1.1', freeze_params=True,
        notes='Best for PubMed + PMC text'),
    'bluebert': EncoderConfig(
        'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        freeze_params=False),
    'pubmedbert': EncoderConfig(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'),
    'roberta_clin': EncoderConfig(
        'roberta-base', freeze_params=False),
    'deberta_med': EncoderConfig(
        'microsoft/deberta-v3-base', freeze_params=False,
        notes='Uses disentangled attention'),
}
