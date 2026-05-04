# LLM from Scratch

## Tokenizing 
I use tiktoken but here's my v1 code
```
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

```

The Tokenizer does was it literally is named. Tokenizes words so that we can use it algebraiclly. 
* We are using a dictionary (imported as vocab)
    * str_to_int is the string to an int and int_to_str is its reverse
* Encode
    * forget the preprocess for this explaination
    * ids is just the ID of the specific word string that was given 
        * A = 1, apple = 2, is = 3, red = 4
    * We return the this encoded version of the text
* Decode
    * the reverse of encode so that it can create text again 


## Embedding 
I use `torch.nn.Embeddings` but the way it works 
* There is a weight matrix of the shape (`numer_of_embeddings`, and `embedding_dimension`)
    * this matrix will contain all the token IDs. Each ID is going to be a 3D vector
    * We can update the weights of the embedding (by back propagation) based on some loss determined when searching for context

## Attention 
This is pretty important and differentiates LLMs from deep learning 
```
class SelfAttention_v2(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
```
* We need three vectors (query, key, value) where query is user query, key is the ID of string, and value is the value of the string 
* We then obtain the attention scores by `input * input.T`
    * you can see that I use `queries @ keys.T` in my code
* We need to find teh softmax to normalize values 
* and then we can finally get the context vector which will emphasize relevance based on query 

### casual attention
* This is what I was talking about yesterday. If we have `input * input.T` the top triangle is going to match the bottom. Why waste the space...
* we can do the following using `torch.triu` to make the top triangle of the matrix `-inf`. Since softmax will become 0
```
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    # finally apply softmax 
    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
```