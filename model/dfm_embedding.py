import torch

from layer import FeaturesEmbedding


class DeepFactorizationMachineModel_embedding(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        #embed_x = embed_x.view(-1, self.embed_output_dim)
        #x = self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return embed_x
