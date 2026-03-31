import torch
import tiktoken
from model_config import batch_size, block_size, device

with open('./words.txt', 'r', encoding='utf-8') as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)[:200_000_000]
data = torch.tensor(tokens, dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)