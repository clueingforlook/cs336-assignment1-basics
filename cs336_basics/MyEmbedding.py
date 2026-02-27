import torch
from torch import nn

class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters:

        Parameters:
            - num_embeddings: int Size of the vocabulary
            - embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            - device: torch.device | None = None Device to store the parameters on
            - dtype: torch.dtype | None = None Data type of the parameters
        """

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype))

        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding module. This function should accept the following parameters:

        Parameters:
            - token_ids: torch.Tensor input tensor of shape (batch_size, seq_length) containing token IDs

        Returns:
            - torch.Tensor output tensor of shape (batch_size, seq_length, embedding_dim) containing the corresponding embedding vectors
        """
        token_ids = token_ids.long()
        return self.weight[token_ids]
        