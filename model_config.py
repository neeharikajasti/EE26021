import torch

batch_size    = 32
block_size    = 128
max_iters     = 200_000
eval_interval = 1_000
learning_rate = 3e-4
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd        = 384
n_head        = 6
n_layer       = 6
dropout       = 0.1