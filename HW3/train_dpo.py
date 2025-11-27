import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
import numpy as np
import os
import pickle
from heuristic import get_diversity_score

# --- Hyperparameters ---
beta = 0.1              # The DPO temperature (strength of preference)
learning_rate = 1e-5    # Very low LR for stability
max_iters = 200         # Short run is enough to see changes
batch_size = 16
block_size = 64         # Must match your trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_dir = 'out-dpo'
os.makedirs(out_dir, exist_ok=True)

# --- Load Data ---
data_dir = 'data/sherlock'
# Check if data exists
if not os.path.exists(os.path.join(data_dir, 'train.bin')):
    raise FileNotFoundError("train.bin not found. Did you run prepare_sherlock.py?")

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

def decode(ids):
    return ''.join([itos.get(i, '') for i in ids])

def get_preference_batch():
    """
    Generates pairs: (Winner, Loser)
    Winner = Higher heuristic score (More valid English)
    Loser = Lower heuristic score (More gibberish)
    """
    data = train_data
    yw_list, yl_list = [], []
    
    while len(yw_list) < batch_size:
        # Sample two random chunks
        idx = torch.randint(len(data) - block_size, (2,))
        chunk1 = data[idx[0]:idx[0]+block_size].astype(np.int64)
        chunk2 = data[idx[1]:idx[1]+block_size].astype(np.int64)
        
        text1 = decode(chunk1.tolist())
        text2 = decode(chunk2.tolist())
        
        s1 = get_diversity_score(text1)
        s2 = get_diversity_score(text2)
        
        # Only train if there is a meaningful difference
        if abs(s1 - s2) < 0.1: 
            continue
            
        if s1 > s2:
            yw_list.append(torch.from_numpy(chunk1))
            yl_list.append(torch.from_numpy(chunk2))
        else:
            yw_list.append(torch.from_numpy(chunk2))
            yl_list.append(torch.from_numpy(chunk1))

    return torch.stack(yw_list).to(device), torch.stack(yl_list).to(device)

def get_batch_logps(model, x):
    """
    Compute log probabilities of the sequence x under the model.
    """
    # FIX: We pass 'targets=x' (dummy targets) to force the model to 
    # compute logits for the ENTIRE sequence, not just the last token.
    logits, _ = model(x, targets=x)
    
    # Shift for autoregressive loss: predict t+1 given t
    # input_ids = x[:, :-1] # Not used explicitly, but logic is implied
    target_ids = x[:, 1:]
    
    # Now logits has shape [B, T, V], so slicing works correctly
    logits = logits[:, :-1, :]
    
    # Log Softmax
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather the log prob of the *actual* token that appeared
    per_token_logps = torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
    
    # Sum log probs over the sequence
    return per_token_logps.sum(-1)

# --- Models ---
# Config MUST match your trained baseline (Sherlock setup)
config = GPTConfig(
    block_size=64, 
    vocab_size=meta['vocab_size'], 
    n_layer=4, n_head=4, n_embd=128, 
    dropout=0.0, 
    bias=False # <--- Crucial: Matches baseline training
)

def load_checkpoint_safe(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model']
    
    # Fix for _orig_mod prefix if model was compiled
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)

print("Loading Policy and Reference Models...")
ckpt_path = 'out-sherlock/ckpt.pt'

# Policy Model (The one we train)
policy_model = GPT(config)
load_checkpoint_safe(policy_model, ckpt_path)
policy_model.to(device)

# Reference Model (Frozen copy)
ref_model = GPT(config)
load_checkpoint_safe(ref_model, ckpt_path)
ref_model.to(device)
ref_model.eval()

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

# --- Training Loop ---
print("Starting DPO Alignment...")
policy_model.train()

for iter in range(max_iters):
    # 1. Get Data
    yw, yl = get_preference_batch()
    
    # 2. Forward Policy
    pi_yw_logps = get_batch_logps(policy_model, yw)
    pi_yl_logps = get_batch_logps(policy_model, yl)
    
    # 3. Forward Reference (No Grad)
    with torch.no_grad():
        ref_yw_logps = get_batch_logps(ref_model, yw)
        ref_yl_logps = get_batch_logps(ref_model, yl)
    
    # 4. DPO Loss
    # Equation: -log sigmoid( beta * ( (pi_w - ref_w) - (pi_l - ref_l) ) )
    pi_ratio = pi_yw_logps - pi_yl_logps
    ref_ratio = ref_yw_logps - ref_yl_logps
    logits = pi_ratio - ref_ratio
    
    loss = -F.logsigmoid(beta * logits).mean()
    
    # 5. Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if iter % 10 == 0:
        # Accuracy: How often does the policy prefer the winner more than the ref?
        acc = (logits > 0).float().mean()
        print(f"Iter {iter}: Loss {loss.item():.4f}, Reward Acc {acc.item():.2f}")

# Save
torch.save(policy_model.state_dict(), os.path.join(out_dir, 'aligned_dpo_model.pt'))
print(f"DPO Alignment Complete. Model saved to {out_dir}/aligned_dpo_model.pt")