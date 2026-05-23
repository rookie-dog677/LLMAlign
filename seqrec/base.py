import torch.nn as nn
import torch

class AbstractModel(nn.Module):
    def __init__(
        self,
        config: dict,
    ):
        super(AbstractModel, self).__init__()
        self.config = config
        self._eval_item_embedding_cache = None

    @property
    def n_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'Total number of trainable parameters: {total_params}'

    def calculate_loss(self, batch):
        raise NotImplementedError('calculate_loss method must be implemented.')

    def predict(self, batch, n_return_sequences=1):
        raise NotImplementedError('predict method must be implemented.')

    def get_embeddings(self, items):
        raise NotImplementedError('get item_embeddings must be implemented.')

    def clear_eval_item_embedding_cache(self):
        self._eval_item_embedding_cache = None
        item_embeddings = getattr(self, 'item_embeddings', None)
        if hasattr(item_embeddings, '_eval_cache'):
            item_embeddings._eval_cache = None

    def train(self, mode: bool = True):
        if mode:
            self.clear_eval_item_embedding_cache()
        return super().train(mode)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.clear_eval_item_embedding_cache()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _get_cached_all_embeddings(self, compute_fn, device=None):
        use_cache = device is not None and not self.training and not torch.is_grad_enabled()
        if not use_cache:
            return compute_fn()

        target_device = torch.device(device)
        cached = self._eval_item_embedding_cache
        if cached is not None and cached.device == target_device:
            item_embeddings = getattr(self, 'item_embeddings', None)
            if hasattr(item_embeddings, '_eval_cache'):
                item_embeddings._eval_cache = cached
            return cached

        all_embeddings = compute_fn()
        if all_embeddings.device != target_device:
            all_embeddings = all_embeddings.to(target_device)

        # Cache a detached tensor because this path is only used under no_grad() in eval.
        self._eval_item_embedding_cache = all_embeddings.detach()
        item_embeddings = getattr(self, 'item_embeddings', None)
        if hasattr(item_embeddings, '_eval_cache'):
            item_embeddings._eval_cache = self._eval_item_embedding_cache
        return self._eval_item_embedding_cache

    def prepare_eval_item_embeddings(self, device=None):
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                return None
        with torch.no_grad():
            return self.get_all_embeddings(device=device)
