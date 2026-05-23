import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class Weight:
    data: object

class Embedding2(nn.Module):
    def __init__(self, adapter, embedding):
        super().__init__()
        self.embedding = embedding
        self.adapter = adapter
        self._eval_cache = None

    def forward(self, indices):
        cache = self._eval_cache
        if (
            cache is not None
            and not self.training
            and not torch.is_grad_enabled()
            and cache.device == indices.device
        ):
            return cache[indices]
        return self.adapter(self.embedding(indices))

    def can_reuse_all_embeddings(self):
        if bool(getattr(self.adapter, 'noisy_gating', False)):
            return False
        if float(getattr(self.adapter, 'expert_dropout', 0.0)) > 0.0:
            return False
        return not any(
            isinstance(module, nn.Dropout) and module.p > 0.0
            for module in self.adapter.modules()
        )

    @property
    def weight(self):
        return Weight(self.adapter(self.embedding.weight.data))
