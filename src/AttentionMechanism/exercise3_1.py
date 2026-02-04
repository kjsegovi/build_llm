from SelfAttention import SelfAttention_v1, SelfAttention_v2
import torch


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

    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))

    sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
    sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
    sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)

    print(sa_v1(inputs))



if __name__ == "__main__":
    main()