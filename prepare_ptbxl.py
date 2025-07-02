import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configurar caminhos
data_path = Path(r"D:\ptb-xl\npy_lr")
output_path = Path("data")
output_path.mkdir(exist_ok=True)

# Carregar metadados do PTB-XL
# Baixe o ptbxl_database.csv do repositório oficial se não tiver
metadata = pd.read_csv(r"D:\ptb-xl\ptbxl_database.csv")

# Dividir em train/val/test usando split recomendado
train_df = metadata[metadata['strat_fold'] <= 8]
val_df = metadata[metadata['strat_fold'] == 9]
test_df = metadata[metadata['strat_fold'] == 10]

# Salvar índices
train_df.to_csv(output_path / "train_records.csv", index=False)
val_df.to_csv(output_path / "val_records.csv", index=False)
test_df.to_csv(output_path / "test_records.csv", index=False)

print(f"Train: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")

# Criar arquivo records.csv consolidado
records_df = metadata.copy()
records_df['split'] = 'train'
records_df.loc[records_df['strat_fold'] == 9, 'split'] = 'val'
records_df.loc[records_df['strat_fold'] == 10, 'split'] = 'test'
records_df['record_id'] = records_df['filename_lr'].str.replace('.hea', '')
records_df.to_csv(output_path / "records.csv", index=False)