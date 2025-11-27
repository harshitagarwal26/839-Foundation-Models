import os
import requests
import numpy as np
import pickle

# Configuration
input_file_path = os.path.join(os.path.dirname(__file__), 'data/sherlock/input.txt')
data_dir = os.path.dirname(input_file_path)
os.makedirs(data_dir, exist_ok=True)

# 1. Download the dataset (Sherlock Holmes from Project Gutenberg)
print("Downloading Sherlock Holmes...")
url = "https://www.gutenberg.org/files/1661/1661-0.txt"
response = requests.get(url)
data = response.text

# Save raw text
with open(input_file_path, 'w', encoding='utf-8') as f:
    f.write(data)

print(f"Length of dataset in characters: {len(data):,}")

# 2. Get unique characters (Vocabulary)
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# Create mapping
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s] 

def decode(l):
    return ''.join([itos[i] for i in l])

# 3. Encode and Save Binaries
train_data = data[:int(len(data)*0.9)]
val_data = data[int(len(data)*0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

# Save meta for the model to use correct vocab size
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Dataset prepared in data/sherlock/")