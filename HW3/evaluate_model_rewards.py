import torch
import matplotlib.pyplot as plt
from model import GPT, GPTConfig
from train_reward_model import RewardGPT
from heuristic import get_diversity_score
import os
import pickle
import textwrap

# --- Config ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dir = 'out-sherlock' 
data_dir = 'data/sherlock'
ckpt_path = os.path.join(model_dir, 'ckpt.pt')
num_samples = 20 # Increased samples for better plot
max_new_tokens = 100

# --- Load Meta ---
if os.path.exists(os.path.join(data_dir, 'meta.pkl')):
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
else:
    stoi, itos = {}, {}
decode = lambda l: ''.join([itos.get(i, '') for i in l])

# --- Load Models ---
print("Loading models...")
# Generator (bias=False)
config = GPTConfig(block_size=64, vocab_size=meta.get('vocab_size', 65), 
                   n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False)
generator = GPT(config)
state_dict = torch.load(ckpt_path, map_location=device)['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
generator.load_state_dict(state_dict)
generator.to(device)
generator.eval()

# Reward Model (bias=True)
rm_config = GPTConfig(block_size=64, vocab_size=meta.get('vocab_size', 65), 
                      n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=True)
reward_model = RewardGPT(rm_config)
rm_path = 'out-reward-model-sherlock/reward_model.pt'
if not os.path.exists(rm_path): rm_path = 'out-reward-model/reward_model.pt'
reward_model.load_state_dict(torch.load(rm_path, map_location=device))
reward_model.to(device)
reward_model.eval()

# --- Generate & Score ---
print(f"Generating {num_samples} samples for analysis...")
results = []
pred_scores = []
true_scores = []

for i in range(num_samples):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = generator.generate(context, max_new_tokens=max_new_tokens, top_k=10)
    text = decode(generated_ids[0].tolist())
    
    true_s = get_diversity_score(text)
    
    rm_input = generated_ids[:, :64] if generated_ids.size(1) >= 64 else generated_ids
    with torch.no_grad():
        pred_s = reward_model(rm_input).item()
        
    results.append({"text": text, "true": true_s, "pred": pred_s})
    pred_scores.append(pred_s)
    true_scores.append(true_s)

# --- Visualization 1: Scatter Plot ---
plt.figure(figsize=(10, 6))
plt.scatter(true_scores, pred_scores, alpha=0.7, color='blue')
plt.title('Reward Model Accuracy: Predicted vs True Heuristic')
plt.xlabel('Ground Truth Score (Heuristic)')
plt.ylabel('Predicted Score (Reward Model)')
plt.grid(True, linestyle='--', alpha=0.6)
# Save plot
plt.savefig('reward_correlation.png')
print("Plot saved to 'reward_correlation.png'")

# --- Visualization 2: Readable Text Log ---
results.sort(key=lambda x: x['pred'], reverse=True)
with open('reward_samples.txt', 'w') as f:
    f.write("=== HIGH REWARD SAMPLES (Top 3) ===\n")
    for i in range(3):
        f.write(f"\n[Rank {i+1}] Pred: {results[i]['pred']:.3f} | True: {results[i]['true']:.3f}\n")
        f.write(textwrap.fill(results[i]['text'], width=80))
        f.write("\n" + "-"*40 + "\n")
        
    f.write("\n\n=== LOW REWARD SAMPLES (Bottom 3) ===\n")
    for i in range(1, 4):
        item = results[-i]
        f.write(f"\n[Rank {num_samples-i+1}] Pred: {item['pred']:.3f} | True: {item['true']:.3f}\n")
        f.write(textwrap.fill(item['text'], width=80))
        f.write("\n" + "-"*40 + "\n")

print("Full text samples saved to 'reward_samples.txt'")