import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # better inheritence properties (so that when VectorQuantizer1d() inherits it, only these will be
        # changed)
        self.permute_order_in = [0, 2, 3, 1]
        self.permute_order_out = [0, 3, 1, 2]

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        2d: z.shape = (batch, channel, height, width)
        1d: z.shape = (batch, channel, time)
        quantization pipeline:
            1. get encoder input 2d: (B,C,H,W) or 1d: (B, C, T)
            2. flatten input to 2d: (B*H*W,C) or 1d: (B*T, C)
        """
        # reshape z -> (batch, height, width, channel) or (batch, time, channel) and flatten
        z = z.permute(self.permute_order_in).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(self.permute_order_out).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(self.permute_order_out).contiguous()

        return z_q

class VectorQuantizer1d(VectorQuantizer):

    def __init__(self, n_embed, embed_dim, beta=0.25):
        super().__init__(n_embed, embed_dim, beta)
        self.permute_order_in = [0, 2, 1]
        self.permute_order_out = [0, 2, 1]


if __name__ == '__main__':
    quantize = VectorQuantizer1d(n_embed=1024, embed_dim=256, beta=0.25)

    # 1d Input (features)
    enc_outputs = torch.rand(6, 256, 53)
    quant, emb_loss, info = quantize(enc_outputs)
    print(quant.shape)

    quantize = VectorQuantizer(n_e=1024, e_dim=256, beta=0.25)

    # Audio
    enc_outputs = torch.rand(4, 256, 5, 53)
    quant, emb_loss, info = quantize(enc_outputs)
    print(quant.shape)

    # Image
    enc_outputs = torch.rand(4, 256, 16, 16)
    quant, emb_loss, info = quantize(enc_outputs)
    print(quant.shape)
