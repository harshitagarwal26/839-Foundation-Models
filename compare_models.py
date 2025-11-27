import torch
import matplotlib.pyplot as plt
from model import GPT, GPTConfig
from heuristic import get_diversity_score
import os
import pickle
import textwrap

# --- Config ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = 'data/sherlock'
baseline_path = 'out-sherlock/ckpt.pt'
aligned_path = 'out-dpo/aligned_dpo_model.pt'
num_samples = 50  
max_tokens = 100

# --- Load Meta ---
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']
decode = lambda l: ''.join([itos.get(i, '') for i in l])

# --- Loader ---
def load_model(path):
    config = GPTConfig(block_size=64, vocab_size=meta['vocab_size'], 
                       n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False)
    model = GPT(config)
    
    # Load Checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # FIX: Check if it's a dict with 'model' key or just the raw state_dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        sd = checkpoint['model']
    else:
        sd = checkpoint # The checkpoint itself is the state dict
        
    # Clean up _orig_mod prefix
    unwanted_prefix = '_orig_mod.'
    for k,v in list(sd.items()):
        if k.startswith(unwanted_prefix): 
            sd[k[len(unwanted_prefix):]] = sd.pop(k)
            
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model

print("Loading models...")
baseline = load_model(baseline_path)
aligned = load_model(aligned_path)

# --- Generate Stats ---
print(f"Generating {num_samples} samples per model for statistics...")
base_scores = []
align_scores = []
comparisons = []

for i in range(num_samples):
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Baseline
    b_ids = baseline.generate(ctx, max_new_tokens=max_tokens, top_k=10)
    b_text = decode(b_ids[0].tolist())
    b_score = get_diversity_score(b_text)
    base_scores.append(b_score)
    
    # Aligned
    a_ids = aligned.generate(ctx, max_new_tokens=max_tokens, top_k=10)
    a_text = decode(a_ids[0].tolist())
    a_score = get_diversity_score(a_text)
    align_scores.append(a_score)
    
    # Save first 5 for text log
    if i < 5:
        comparisons.append((b_text, b_score, a_text, a_score))

# --- Visualization 1: Histogram ---
plt.figure(figsize=(10, 6))
plt.hist(base_scores, bins=15, alpha=0.5, label='Baseline', color='blue')
plt.hist(align_scores, bins=15, alpha=0.5, label='Aligned (DPO)', color='orange')
plt.xlabel('Heuristic Score (Valid English Ratio)')
plt.ylabel('Frequency')
plt.title('Impact of DPO Alignment on Model Quality')
plt.legend(loc='upper right')
plt.savefig('alignment_improvement.png')
print("Histogram saved to 'alignment_improvement.png'")

# --- Visualization 2: Readable Log ---
with open('alignment_comparison.txt', 'w') as f:
    f.write(f"Average Baseline Score: {sum(base_scores)/len(base_scores):.4f}\n")
    f.write(f"Average Aligned Score:  {sum(align_scores)/len(align_scores):.4f}\n")
    f.write("="*80 + "\n\n")
    
    for i, (b_txt, b_sc, a_txt, a_sc) in enumerate(comparisons):
        f.write(f"SAMPLE {i+1}\n")
        f.write(f"--- Baseline (Score: {b_sc:.2f}) ---\n")
        f.write(textwrap.fill(b_txt, width=80))
        f.write(f"\n\n--- Aligned (Score: {a_sc:.2f}) ---\n")
        f.write(textwrap.fill(a_txt, width=80))
        f.write("\n" + "="*80 + "\n")

print("Text comparison saved to 'alignment_comparison.txt'")