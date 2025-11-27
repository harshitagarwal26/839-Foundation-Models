import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT, GPTConfig
import numpy as np
import os
import pickle
from heuristic import get_diversity_score

# --- Shared Configuration & Globals ---
# These need to be global so evaluate_model_rewards.py can import them
out_dir = 'out-reward-model-sherlock' # New output folder
max_iters = 500 
learning_rate = 3e-4
batch_size = 32
block_size = 64 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(out_dir, exist_ok=True)

# --- Data Loader Globals ---
data_dir = 'data/sherlock'            # New data
# We use a try-except block or check existence to prevent errors if data isn't ready
if os.path.exists(os.path.join(data_dir, 'meta.pkl')):
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
else:
    # Fallback if imported without data preparation (sanity check)
    stoi, itos = {}, {}

def decode(ids):
    return ''.join([itos.get(i, '') for i in ids])

# --- Model Definition ---
class RewardGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gpt = GPT(config)
        # Replace final head to output 1 scalar instead of vocab_size
        self.gpt.lm_head = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
        # Forward through transformer body
        tok_emb = self.gpt.transformer.wte(idx)
        pos_emb = self.gpt.transformer.wpe(pos)
        x = self.gpt.transformer.drop(tok_emb + pos_emb)
        for block in self.gpt.transformer.h:
            x = block(x)
        x = self.gpt.transformer.ln_f(x)
        
        # Pool: Average of all tokens to get summary of text style
        x_avg = x.mean(dim=1) 
        reward = self.gpt.lm_head(x_avg)
        return reward

# --- Training Logic (Protected) ---
# This block ONLY runs if you execute "python train_reward_model.py"
# It will NOT run if you import from this file.
if __name__ == "__main__":
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    def get_batch():
        data = train_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        
        # Calculate ground truth rewards (diversity score)
        rewards = []
        for seq in x:
            text = decode(seq.tolist())
            rewards.append(get_diversity_score(text))
        
        # Target shape (B, 1)
        y = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)
        x = x.to(device)
        return x, y

    # Small config for the reward model
    config = GPTConfig(
        block_size=block_size, vocab_size=meta['vocab_size'], 
        n_layer=4, n_head=4, n_embd=128, dropout=0.0
    )
    model = RewardGPT(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Training Reward Model...")
    for iter in range(max_iters):
        xb, yb = get_batch()
        pred_rewards = model(xb)
        loss = F.mse_loss(pred_rewards, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"Iter {iter}: Loss {loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(out_dir, 'reward_model.pt'))
    print("Reward Model Saved.")