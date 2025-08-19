import torch 
import torch.nn as nn
import math



class TimeEmbed(nn.Module):
    """ 
        This class computes time embedding (positional embedding) of time stamps with traditional mechanics from Transformer architecture
            (b, ) -> (b, embd_dim)
    """
    def __init__(self, embed_dim:int, base: int = 10000):
        super().__init__()
        assert embed_dim % 2 == 0, "embedding dimension must be even"
        self.embed_dim = embed_dim 
        # define divisor tensor with e^log(a^b) = a^b for numerical stability
        divisor = torch.exp(
            -math.log(base) * (torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim)
        )
        # register the divisor as non-traninable parameters
        self.register_buffer(name="divisor", tensor=divisor, persistent=False)

    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # expand input dimension: (b, ) -> (b, 1) for broadcasting - this is the numerator
        x = x.to(dtype=torch.float32).unsqueeze(dim=1) 
        # pleaseholder for final tensor
        embedding = torch.zeros(size=(x.shape[0],self.embed_dim), device=x.device, dtype=x.dtype)
        # final value assignment
        embedding[:,::2] = torch.sin(x/self.divisor)
        embedding[:,1::2] = torch.cos(x/self.divisor)
        return embedding
