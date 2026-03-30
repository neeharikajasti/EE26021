import torch
import tiktoken

batch_size = 32
block_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('./words.txt', 'r', encoding='utf-8') as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)

tokens = tokens[:200_000_000]

data = torch.tensor(tokens, dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


data_split = train_data
ix = torch.randint(len(data_split) - block_size, (batch_size,))
x = torch.stack([data_split[i:i+block_size] for i in ix])
y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
x.to(device)
y.to(device)