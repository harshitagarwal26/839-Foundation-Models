import torch
from model import GPT, GPTConfig
from verifier import get_verifier_reward
import os
import pickle
import random

# --- Config ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CHANGE THIS PATH TO SWITCH BETWEEN BASELINE AND ALIGNED
model_path = 'out-sherlock/ckpt.pt' 
# model_path = 'out-grpo/grpo_model.pt'

data_dir = 'data/sherlock'
num_prompts = 50
max_new_tokens = 64

PROMPT_LIST = [
    "The ", "It ", "When ", "If ", "He ", "She ", "But ", "For ", "In ", "On ", 
    "At ", "Who ", "Why ", "How ", "Where ", "My ", "Your ", "Our ", "Their ", "His ",
    "Her ", "They ", "We ", "You ", "I ", "A ", "An ", "Some ", "Many ", "Few ",
    "This ", "That ", "These ", "Those ", "Here ", "There ", "Now ", "Then ", "Once ", "One ",
    "Two ", "Three ", "After ", "Before ", "While ", "During ", "Since ", "Until ", "Because ", "Although "
]

# --- Load Meta ---
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '') for i in l])

# --- Load Model ---
def load_model(path):
    print(f"Loading {path}...")
    checkpoint = torch.load(path, map_location=device)
    config = GPTConfig(
        block_size=64, vocab_size=meta['vocab_size'], 
        n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False
    )
    model = GPT(config)
    sd = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    
    clean_sd = {k.replace('_orig_mod.', ''): v for k,v in sd.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    return model

model = load_model(model_path)

print(f"Evaluating on {num_prompts} prompts...")
total_score = 0
examples = []

for i in range(num_prompts):
    # Iterate through list or sample randomly
    prompt_text = PROMPT_LIST[i % len(PROMPT_LIST)]
    prompt_ids = torch.tensor(encode(prompt_text), dtype=torch.long, device=device).unsqueeze(0)
    
    out_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens, top_k=50)
    out_text = decode(out_ids[0].tolist())
    
    score = get_verifier_reward(out_text)
    total_score += score
    
    if i < 3: 
        examples.append((out_text, score))

print("="*60)
print(f"Mean Verifier Score: {total_score/num_prompts:.4f}")
print("="*60)
for txt, sc in examples:
    clean_txt = txt.replace('\n', ' ')[:70]
    print(f"Score: {sc} | Text: {clean_txt}...")