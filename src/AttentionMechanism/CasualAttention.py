import torch
from SelfAttention import SelfAttention_v2

def main(): 
    inputs = torch.tensor(
        [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],
        ]  # step     (x^6)
    )
    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)

    attn_scores = queries @ keys.T
    attn_weight = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    print(attn_weight)


    # we will use the PyTorch funtion tril to create a mask where the values above the diagonal are zero
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(f"Context Length: {context_length}")
    print(mask_simple)
    
    # now that we have a ones matrix with only the lower triangle we can multiply the mask with the attention 
    # weights to zero-out the triangle above
    masked_simple = mask_simple * attn_weight
    print(masked_simple)

    # we now want to renormalize the attention weights to sum up to 1 again in each row
    # we can do this by dividing each element in the row by the sum of the row
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    print(row_sums)
    masked_simple_norm = masked_simple / row_sums
    print(masked_simple_norm)


    # improving this by setting upper diagonal to -inf 
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)

    # finally apply softmax 
    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
    print(attn_weights)


if __name__ == "__main__":
    main()
