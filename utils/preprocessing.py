import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset

class ClinicalDataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.label_encoders = {}
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def preprocess_timeseries(self, data: pd.DataFrame) -> np.ndarray:
        """Tiền xử lý dữ liệu time-series (vital signs, blood tests)"""
        # Xử lý missing values
        imputer = SimpleImputer(strategy='forward_fill')
        data_imputed = imputer.fit_transform(data)
        
        # Chuẩn hóa dữ liệu
        if 'timeseries' not in self.scalers:
            self.scalers['timeseries'] = StandardScaler()
            data_scaled = self.scalers['timeseries'].fit_transform(data_imputed)
        else:
            data_scaled = self.scalers['timeseries'].transform(data_imputed)
        
        return data_scaled
    
    def preprocess_categorical(self, data: Dict[str, pd.Series]) -> Dict[str, torch.Tensor]:
        """Tiền xử lý dữ liệu categorical"""
        processed_data = {}
        
        for feature_name, series in data.items():
            if feature_name not in self.label_encoders:
                self.label_encoders[feature_name] = LabelEncoder()
                encoded = self.label_encoders[feature_name].fit_transform(series.fillna('unknown'))
            else:
                encoded = self.label_encoders[feature_name].transform(series.fillna('unknown'))
            
            processed_data[feature_name] = torch.tensor(encoded, dtype=torch.long)
        
        return processed_data
    
    def preprocess_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tiền xử lý clinical notes"""
        # Làm sạch text
        cleaned_texts = []
        for text in texts:
            if pd.isna(text) or text == "":
                cleaned_texts.append("No clinical notes available")
            else:
                cleaned_texts.append(str(text).lower().strip())
        
        # Tokenization
        encoding = self.tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=self.config.text_max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
    def create_sequences(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo sequences cho time-series data"""
        sequences = []
        sequence_labels = []
        
        for i in range(len(data) - self.config.sequence_length - self.config.prediction_window + 1):
            seq = data[i:i + self.config.sequence_length]
            label = labels[i + self.config.sequence_length + self.config.prediction_window - 1]
            
            sequences.append(seq)
            sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)

class ClinicalDataset(Dataset):
    def __init__(self, timeseries_data, categorical_data, text_data, labels):
        self.timeseries_data = timeseries_data
        self.categorical_data = categorical_data
        self.text_data = text_data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'timeseries': torch.FloatTensor(self.timeseries_data[idx]),
            'categorical': {k: v[idx] for k, v in self.categorical_data.items()},
            'text_input_ids': self.text_data['input_ids'][idx],
            'text_attention_mask': self.text_data['attention_mask'][idx],
            'labels': torch.FloatTensor([self.labels[idx]])
        }

def load_and_preprocess_data(data_path: str, config) -> Tuple[ClinicalDataset, ClinicalDataset, ClinicalDataset]:
    """Load và preprocess dữ liệu từ file"""
    # Load dữ liệu (giả sử có file CSV)
    df = pd.read_csv(data_path)
    
    # Khởi tạo preprocessor
    preprocessor = ClinicalDataPreprocessor(config)
    
    # Tách các loại dữ liệu
    timeseries_cols = [col for col in df.columns if 'vital_' in col or 'lab_' in col]
    categorical_cols = ['age_group', 'gender', 'admission_type', 'department']
    text_col = 'clinical_notes'
    label_col = 'clinical_deterioration'
    
    # Preprocess từng loại dữ liệu
    timeseries_data = preprocessor.preprocess_timeseries(df[timeseries_cols])
    categorical_data = preprocessor.preprocess_categorical({col: df[col] for col in categorical_cols})
    text_data = preprocessor.preprocess_text(df[text_col].tolist())
    labels = df[label_col].values
    
    # Tạo sequences
    timeseries_sequences, sequence_labels = preprocessor.create_sequences(timeseries_data, labels)
    
    # Chia dữ liệu train/val/test
    from sklearn.model_selection import train_test_split
    
    # Train/temp split
    train_idx, temp_idx = train_test_split(
        range(len(sequence_labels)), 
        test_size=0.4, 
        random_state=config.seed, 
        stratify=sequence_labels
    )
    
    # Val/test split
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.5, 
        random_state=config.seed, 
        stratify=sequence_labels[temp_idx]
    )
    
    # Tạo datasets
    def create_dataset(indices):
        return ClinicalDataset(
            timeseries_sequences[indices],
            {k: v[indices] for k, v in categorical_data.items()},
            {k: v[indices] for k, v in text_data.items()},
            sequence_labels[indices]
        )
    
    train_dataset = create_dataset(train_idx)
    val_dataset = create_dataset(val_idx)
    test_dataset = create_dataset(test_idx)
    
    return train_dataset, val_dataset, test_dataset
