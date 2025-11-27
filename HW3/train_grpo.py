import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import random
import matplotlib.pyplot as plt
import re
import os
import pickle
from model import GPT, GPTConfig 

# -----------------------------------------------------------------------------
# 0. LOGGING SETUP (NEW)
# -----------------------------------------------------------------------------
# This file will contain all your results for the assignment
output_file = "grpo_assignment_results.txt"

# Clear the file at the start of the run
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== GRPO EXPERIMENT RESULTS ===\n\n")

def log(msg):
    """Prints to terminal and appends to the text file."""
    print(msg)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_new_tokens': 32,      
    'group_size': 4,           
    'num_steps': 100,          
    'batch_size': 4,           
    'learning_rate': 1e-5,     
    'beta': 0.04,              
    'clip_eps': 0.2,           
    'chk_path': 'out-sherlock/ckpt.pt' # Ensure this path is correct
}

# -----------------------------------------------------------------------------
# 2. THE VERIFIER
# -----------------------------------------------------------------------------
def verifier_reward(text):
    """Maximize 'e' characters."""
    count = text.count('e') + text.count('E')
    return float(count)

# -----------------------------------------------------------------------------
# 3. DATA
# -----------------------------------------------------------------------------
sherlock_prompts = [
    "The detective looked at", "Sherlock Holmes said", "The evidence suggests",
    "Watson, come here", "The red headed league", "It was a dark",
    "I deduce that", "The game is", "A scandal in", "The blue carbuncle",
    "My dear Watson", "The man was wearing", "He examined the", "The police found",
    "A mysterious case", "The hound of", "Dr. Watson replied", "The door opened",
    "There is no doubt", "The footprint was"
]

def get_batch(batch_size):
    return [random.choice(sherlock_prompts) for _ in range(batch_size)]

# -----------------------------------------------------------------------------
# 4. MODEL LOADING (With Safety Checks)
# -----------------------------------------------------------------------------
log(f"Loading model on {config['device']}...")

# Load Checkpoint
try:
    checkpoint = torch.load(config['chk_path'], map_location=config['device'])
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Fix prefixes
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    
    # SAFETY: Check actual vocab size from weights, not just config
    if hasattr(model.transformer, 'wte'):
        real_vocab_size = model.transformer.wte.weight.shape[0]
    else:
        real_vocab_size = gptconf.vocab_size

    log(f"Loaded Checkpoint. Config Vocab: {gptconf.vocab_size}, Real Vocab: {real_vocab_size}")

except Exception as e:
    log(f"Checkpoint error: {e}. Initializing random model.")
    gptconf = GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=65) 
    model = GPT(gptconf)
    real_vocab_size = 65

model.to(config['device'])

# Reference Model
ref_model = GPT(gptconf)
ref_model.load_state_dict(model.state_dict())
ref_model.to(config['device'])
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# Tokenizer Logic
if real_vocab_size < 1000:
    log("Detected Character-level model.")
    meta_path = 'data/sherlock/meta.pkl' # Adjust if needed
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        log("Meta.pkl not found, using ASCII fallback.")
        encode = lambda s: [ord(c) % real_vocab_size for c in s]
        decode = lambda l: ''.join([chr(i) if i < 128 else '?' for i in l])
else:
    log("Detected BPE model.")
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# -----------------------------------------------------------------------------
# 5. BASELINE EVALUATION
# -----------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
stats = {'rewards': [], 'losses': []}

log("\n------------------------------------------------")
log("Phase 1: Baseline Evaluation (Before Training)")
log("------------------------------------------------")

baseline_rewards = []
model.eval()
for i in range(5): 
    prompt = random.choice(sherlock_prompts)
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=config['device']).unsqueeze(0)
    
    # Safety clamp
    idx = torch.clamp(idx, max=real_vocab_size-1)

    gen = model.generate(idx, max_new_tokens=config['max_new_tokens'], top_k=10)[0].tolist()
    text = decode(gen)
    completion = text[len(prompt):]
    score = verifier_reward(completion)
    baseline_rewards.append(score)
    log(f"[Base {i+1}] Prompt: '{prompt}'")
    log(f"          Output: ...{completion.strip().replace(chr(10), ' ')[:60]}")
    log(f"          Reward: {score}")

mean_baseline = sum(baseline_rewards)/len(baseline_rewards)
log(f"\n>>> Mean Baseline Reward: {mean_baseline:.2f}")

# -----------------------------------------------------------------------------
# 6. GRPO TRAINING LOOP
# -----------------------------------------------------------------------------
log("\n------------------------------------------------")
log("Phase 2: GRPO Training")
log("------------------------------------------------")
model.train()

for step in range(config['num_steps']):
    
    # --- ROLLOUT ---
    prompts = get_batch(config['batch_size'])
    batch_inputs = []
    batch_rewards = []
    
    with torch.no_grad():
        for prompt in prompts:
            p_ids = encode(prompt)
            p_ids = [min(x, real_vocab_size-1) for x in p_ids] # Safety
            idx_cond = torch.tensor(p_ids, dtype=torch.long, device=config['device']).unsqueeze(0)
            
            group_texts = []
            group_rewards = []
            
            for _ in range(config['group_size']):
                gen_idx = model.generate(idx_cond, max_new_tokens=config['max_new_tokens'], temperature=1.0, top_k=20)
                completion_ids = gen_idx[0].tolist()[len(p_ids):]
                completion_text = decode(completion_ids)
                rew = verifier_reward(completion_text)
                
                group_texts.append(completion_ids)
                group_rewards.append(rew)
            
            # Advantage
            g_tensor = torch.tensor(group_rewards)
            mean_r = g_tensor.mean()
            std_r = g_tensor.std() + 1e-8
            advantages = (g_tensor - mean_r) / std_r
            
            for i in range(config['group_size']):
                batch_inputs.append((p_ids, group_texts[i], advantages[i].item()))
                batch_rewards.append(group_rewards[i])

    # --- OPTIMIZATION ---
    optimizer.zero_grad()
    loss_accum = 0
    
    for p_ids, c_ids, adv in batch_inputs:
        full_ids = torch.tensor(p_ids + c_ids, dtype=torch.long, device=config['device']).unsqueeze(0)
        
        # Current Policy
        logits, _ = model(full_ids, targets=full_ids) 
        
        start = len(p_ids) - 1 
        end = len(p_ids) + len(c_ids) - 1
        
        rel_logits = logits[0, start:end, :]
        rel_targets = torch.tensor(c_ids, dtype=torch.long, device=config['device'])
        
        log_probs = F.log_softmax(rel_logits, dim=-1)
        action_log_probs = log_probs.gather(1, rel_targets.unsqueeze(1)).squeeze(1)
        
        # Reference Policy
        with torch.no_grad():
            ref_logits, _ = ref_model(full_ids, targets=full_ids)
            ref_rel_logits = ref_logits[0, start:end, :]
            ref_log_probs = F.log_softmax(ref_rel_logits, dim=-1)
            ref_action_log_probs = ref_log_probs.gather(1, rel_targets.unsqueeze(1)).squeeze(1)

        # Loss
        ratio = torch.exp(action_log_probs - action_log_probs.detach())
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - config['clip_eps'], 1.0 + config['clip_eps']) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        
        kl = action_log_probs - ref_action_log_probs
        kl_loss = config['beta'] * kl.mean()
        
        loss = policy_loss + kl_loss
        loss.backward()
        loss_accum += loss.item()

    optimizer.step()
    
    avg_reward = sum(batch_rewards) / len(batch_rewards)
    stats['rewards'].append(avg_reward)
    
    if step % 10 == 0:
        log(f"Step {step}/{config['num_steps']} | Avg Reward: {avg_reward:.4f} | Loss: {loss_accum:.4f}")

# -----------------------------------------------------------------------------
# 7. FINAL EVALUATION & PLOTS
# -----------------------------------------------------------------------------
log("\n------------------------------------------------")
log("Phase 3: Final Evaluation (After Training)")
log("------------------------------------------------")

final_rewards = []
model.eval()

# 1. Quantitative: Calculate Mean Reward
for i in range(10): # Run more samples for better mean
    prompt = random.choice(sherlock_prompts)
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=config['device']).unsqueeze(0)
    idx = torch.clamp(idx, max=real_vocab_size-1)
    
    gen = model.generate(idx, max_new_tokens=config['max_new_tokens'], top_k=10)[0].tolist()
    text = decode(gen)
    completion = text[len(prompt):]
    score = verifier_reward(completion)
    final_rewards.append(score)
    log(f"[Final {i+1}] Prompt: '{prompt}' | Reward: {score}")

final_mean = sum(final_rewards) / len(final_rewards)
log(f"\n>>> Final Mean Reward: {final_mean:.2f}")
log(f">>> Improvement: {final_mean - mean_baseline:+.2f}")

# 2. Qualitative: Side-by-Side Comparison
log("\n------------------------------------------------")
log("Phase 4: Qualitative Comparison")
log("------------------------------------------------")
test_prompt = "Sherlock Holmes sat"
idx = torch.tensor(encode(test_prompt), dtype=torch.long, device=config['device']).unsqueeze(0)
idx = torch.clamp(idx, max=real_vocab_size-1)

# Ref Generation
ref_gen = ref_model.generate(idx, max_new_tokens=32, top_k=1)[0].tolist()
log(f"Prompt: {test_prompt}")
log(f"BEFORE (Base): {decode(ref_gen)}")

# RLVR Generation
new_gen = model.generate(idx, max_new_tokens=32, top_k=1)[0].tolist()
log(f"AFTER (GRPO):  {decode(new_gen)}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(stats['rewards'], label='Verifier Reward')
plt.axhline(y=mean_baseline, color='r', linestyle='--', label='Baseline Mean')
plt.title('GRPO Training: Reward Evolution')
plt.xlabel('Step')
plt.ylabel('Mean Reward')
plt.legend()
plt.grid(True)
plt.savefig('grpo_rewards.png')
log("\nPlot saved to 'grpo_rewards.png'")
log(f"Detailed logs saved to '{output_file}'")