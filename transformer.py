"""Encoder-only transformer.

Reference material:

https://nlp.seas.harvard.edu/annotated-transformer

"""


class Transformer:
    """
    - Embeddings
    - Positional encoding
    - Encoder
        - MultiheadAttention
        - LayerNorm
        - FeedForward
        - LayerNorm
    """
    def __init__(self):
        raise NotImplementedError

    def self_attention(self):
        raise NotImplementedError

    def embedding(self):
        raise NotImplementedError

    def layer_norm(self):
        raise NotImplementedError

    def positional_encoding(self):
        raise NotImplementedError
