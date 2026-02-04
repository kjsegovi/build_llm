import torch

from src.AttentionMechanism.simplified import attn_scores_2, context_vec_2

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

# we need to init the query, key, and value matricese
# setting to grad = false to reduce clutter but can enable to update matrices during model training
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# compute the query, key, and value vectors
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# we need the key and value vectors for all the input elements to find query q
keys = inputs @ W_key
values = inputs @ W_value
print(f"key shape: {keys.shape}")
print(f"value shape: {values.shape}")

# next we need to compute the attention scores.
# attn_score w_22
keys_2 = keys[1]
attn_scores_22 = query_2.dot(keys_2)
print(attn_scores_22)
# we can generalize this to all attention scores via matrix multiplication
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

# attention scores -> attention weights
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**2, dim=-1)
print(attn_weights_2)

# finally we get the context vector
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)