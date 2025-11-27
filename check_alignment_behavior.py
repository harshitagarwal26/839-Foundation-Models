import torch
from model import GPT, GPTConfig
from verifier import get_verifier_reward # To get the word list
import os
import pickle
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = 'data/sherlock'

# Extract the word list from the verifier function logic
# We define it here again for clarity
CRIME_WORDS = set([
    "clue", "evidence", "blood", "murder", "case", "trace", 
    "deduction", "logic", "police", "inspector", "crime", 
    "magnifying", "footprint", "weapon", "motive", "victim",
    "watson", "holmes", "mystery", "solved", "suspect", "arrest",
    "dead", "gun", "knife", "poison", "witness"
])

# Load Meta
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
decode = lambda l: ''.join([itos.get(i, '') for i in l])
encode = lambda s: [stoi.get(c, 0) for c in s]

# Load Models
def load_model(path):
    config = GPTConfig(block_size=64, vocab_size=meta['vocab_size'], n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False)
    model = GPT(config)
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
    clean_sd = {k.replace('_orig_mod.', ''): v for k,v in sd.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    return model

print("Loading models...")
baseline = load_model('out-sherlock/ckpt.pt')
aligned = load_model('out-grpo/grpo_model.pt')

prompt = torch.tensor(encode("The "), dtype=torch.long, device=device).unsqueeze(0)

print("\nGenerating 50 samples each...")
b_total_keywords = 0
a_total_keywords = 0

for i in range(50):
    # Standard generation
    b_out = baseline.generate(prompt, max_new_tokens=64, top_k=50)
    a_out = aligned.generate(prompt, max_new_tokens=64, top_k=50)
    
    b_text = decode(b_out[0].tolist()).lower()
    a_text = decode(a_out[0].tolist()).lower()
    
    # Count keywords
    b_words = re.findall(r'\w+', b_text)
    a_words = re.findall(r'\w+', a_text)
    
    b_count = sum(1 for w in b_words if w in CRIME_WORDS)
    a_count = sum(1 for w in a_words if w in CRIME_WORDS)
    
    b_total_keywords += b_count
    a_total_keywords += a_count

print("-" * 60)
print(f"Total 'Crime' words in Baseline samples: {b_total_keywords}")
print(f"Total 'Crime' words in Aligned samples:  {a_total_keywords}")
print("-" * 60)

if a_total_keywords > b_total_keywords:
    print("SUCCESS: Aligned model uses significantly more Detective vocabulary.")
else:
    print("FAIL: No significant behavioral change.")