import torch
import torch.nn as nn
import numpy as np
from seqrec.base import AbstractModel
from seqrec.modules import TransformerEncoder_v2, get_attention_mask, gather_indexes, MoEAdaptorLayer
from ..Embedding2 import Embedding2


class BERT4Rec(AbstractModel):
    """BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformers.

    Training: randomly mask items in the sequence and predict them (Masked Item Prediction).
    Inference: mask the last position and predict the next item.
    """

    def __init__(self, config: dict, pretrained_item_embeddings=None):
        super(BERT4Rec, self).__init__(config=config)
        self.config = config
        self.mask_ratio = config.get('mask_ratio', 0.2)
        self.mask_token_id = config['item_num'] + 1  # special [MASK] token
        self.uses_pretrained_item_embeddings = pretrained_item_embeddings is not None

        self.load_item_embeddings(pretrained_item_embeddings)

        # Learnable embedding for [MASK] token
        self.mask_embedding = nn.Parameter(torch.empty(1, config['hidden_size']))
        nn.init.normal_(self.mask_embedding, 0, 0.02)

        # Positional embeddings
        self.positional_embeddings = nn.Embedding(
            num_embeddings=config['max_seq_length'],
            embedding_dim=config['hidden_size']
        )
        pos_init_std = 1.0 if self.uses_pretrained_item_embeddings else 0.02
        nn.init.normal_(self.positional_embeddings.weight, 0, pos_init_std)

        self.use_embedding_layer_norm = bool(config.get('embedding_layer_norm', False))
        self.LayerNorm = (
            nn.LayerNorm(config['hidden_size'], eps=1e-12)
            if self.use_embedding_layer_norm
            else nn.Identity()
        )
        self.emb_dropout = nn.Dropout(config['dropout'])

        # Bidirectional Transformer encoder
        self.config['inner_size'] = int(self.config.get('inner_size', self.config['hidden_size'] * 4))
        self.transformer_encoder = TransformerEncoder_v2(self.config)

        # Loss
        if config['loss_type'] == 'bce':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif config['loss_type'] == 'ce':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {config['loss_type']}")

    def load_item_embeddings(self, pretrained_embs):
        if pretrained_embs is None:
            self.item_embeddings = nn.Embedding(
                num_embeddings=self.config['item_num'] + 1,
                embedding_dim=self.config['hidden_size'],
                padding_idx=0
            )
            nn.init.normal_(self.item_embeddings.weight, 0, 0.02)
            with torch.no_grad():
                self.item_embeddings.weight[0].fill_(0)
        else:
            more_token = 0
            assert pretrained_embs.shape[0] == self.config['item_num'] + 1
            self.pretrained_item_embeddings = nn.Embedding.from_pretrained(
                torch.cat([
                    pretrained_embs,
                    torch.randn(more_token, pretrained_embs.shape[-1]).to(pretrained_embs.device)
                ]),
                padding_idx=0
            )
            self.pretrained_item_embeddings.weight.requires_grad = False
            if more_token > 0:
                self.pretrained_item_embeddings.weight[-more_token:].requires_grad = True

            adapter_type = str(self.config.get('item_adapter_type', 'linear')).lower()
            if adapter_type == 'linear':
                assert self.config['adapter_dims'][-1] == -1
                mlp_dims = [self.pretrained_item_embeddings.embedding_dim] + self.config['adapter_dims']
                mlp_dims[-1] = self.config['hidden_size']

                self.item_embeddings_adapter = nn.Sequential()
                self.item_embeddings_adapter.add_module('linear_0', nn.Linear(mlp_dims[0], mlp_dims[1]))
                for i in range(1, len(mlp_dims) - 1):
                    self.item_embeddings_adapter.add_module(f'activation_{i}', nn.ReLU())
                    self.item_embeddings_adapter.add_module(f'linear_{i}', nn.Linear(mlp_dims[i], mlp_dims[i + 1]))

                for name, param in self.item_embeddings_adapter.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif adapter_type == 'moe':
                moe_n_exps = int(self.config.get('moe_n_exps', 8))
                if moe_n_exps < 1:
                    raise ValueError(f"moe_n_exps must be >=1, got {moe_n_exps}")
                moe_dropout = float(self.config.get('moe_dropout', self.config.get('dropout', 0.0)))
                moe_noise = bool(self.config.get('moe_noise', False))
                moe_top_k = int(self.config.get('moe_top_k', 2))
                moe_temperature = float(self.config.get('moe_temperature', 1.0))
                self.item_embeddings_adapter = MoEAdaptorLayer(
                    n_exps=moe_n_exps,
                    layers=[self.pretrained_item_embeddings.embedding_dim, self.config['hidden_size']],
                    dropout=moe_dropout,
                    noise=moe_noise,
                    top_k=moe_top_k,
                    gate_temperature=moe_temperature,
                )
            elif adapter_type == 'llmemb_bottleneck':
                emb_dim = self.pretrained_item_embeddings.embedding_dim
                self.item_embeddings_adapter = nn.Sequential(
                    nn.Linear(emb_dim, emb_dim // 2),
                    nn.Linear(emb_dim // 2, self.config['hidden_size'])
                )
                for m in self.item_embeddings_adapter:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        nn.init.constant_(m.bias, 0)
            elif adapter_type in {'none', 'identity'}:
                emb_dim = self.pretrained_item_embeddings.embedding_dim
                hidden_size = self.config['hidden_size']
                if emb_dim != hidden_size:
                    raise ValueError(
                        f"item_adapter_type={adapter_type} requires embedding_dim == hidden_size, "
                        f"got {emb_dim} vs {hidden_size}"
                    )
                self.item_embeddings_adapter = nn.Identity()
            else:
                raise ValueError(f"Unsupported item_adapter_type: {adapter_type}")

            self.item_embeddings = Embedding2(self.item_embeddings_adapter, self.pretrained_item_embeddings)

    def get_embeddings(self, items):
        """Get embeddings for item ids, handling [MASK] token specially."""
        return self._get_item_embeddings(items)

    def _get_item_embeddings(self, items, all_item_embeddings=None):
        mask_positions = (items == self.mask_token_id)
        safe_items = items.clamp(max=self.config['item_num'])
        if all_item_embeddings is not None:
            embs = all_item_embeddings[safe_items]
        else:
            embs = self.item_embeddings(safe_items)
        # Replace mask token embeddings without clone or .any() CUDA sync
        embs = torch.where(
            mask_positions.unsqueeze(-1),
            self.mask_embedding.expand_as(embs),
            embs,
        )
        return embs

    def _compute_all_embeddings(self):
        if isinstance(self.item_embeddings, nn.Embedding):
            return self.item_embeddings.weight
        return self.item_embeddings.weight.data

    def get_all_embeddings(self, device=None):
        return self._get_cached_all_embeddings(self._compute_all_embeddings, device=device)

    def _can_reuse_all_item_embeddings(self):
        return (
            hasattr(self.item_embeddings, 'can_reuse_all_embeddings')
            and self.item_embeddings.can_reuse_all_embeddings()
        )

    def _mask_sequence(self, item_seqs):
        """Randomly mask items in the sequence for training (vectorized)."""
        masked_seqs = item_seqs.clone()
        mask_labels = torch.zeros_like(item_seqs)

        non_pad_mask = (item_seqs != 0)
        should_mask = (torch.rand(item_seqs.shape, device=item_seqs.device) < self.mask_ratio) & non_pad_mask

        mask_labels[should_mask] = item_seqs[should_mask]
        masked_seqs[should_mask] = self.mask_token_id

        return masked_seqs, mask_labels

    def get_representation(self, item_seqs, all_item_embeddings=None):
        """Encode a (possibly masked) sequence with bidirectional Transformer."""
        inputs_emb = self._get_item_embeddings(item_seqs, all_item_embeddings=all_item_embeddings)
        inputs_emb += self.positional_embeddings(
            torch.arange(self.config['max_seq_length']).to(inputs_emb.device)
        )
        seq = self.emb_dropout(self.LayerNorm(inputs_emb))

        # Bidirectional attention mask (mask only padding positions)
        mask = torch.ne(item_seqs, 0).float().to(inputs_emb.device)
        # Also treat mask_token_id as valid (non-padding)
        mask = torch.where(item_seqs == self.mask_token_id, torch.ones_like(mask), mask)
        attention_mask = get_attention_mask(mask, bidirectional=True)

        seq = self.transformer_encoder(seq, attention_mask=attention_mask)
        return seq[-1]  # last layer output

    def forward(self, batch):
        item_seqs = batch['item_seqs'].clone()
        labels = batch['labels']
        seq_lengths = batch['seq_lengths']
        max_len = self.config['max_seq_length']

        if not isinstance(seq_lengths, torch.Tensor):
            seq_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=item_seqs.device)

        batch_idx = torch.arange(item_seqs.size(0), device=item_seqs.device)

        # Put target item back into the sequence for Cloze training
        can_place = seq_lengths < max_len
        if can_place.any():
            item_seqs[batch_idx[can_place], seq_lengths[can_place]] = labels[can_place]
        full_mask = ~can_place
        if full_mask.any():
            item_seqs[full_mask] = torch.roll(item_seqs[full_mask], -1, dims=1)
            item_seqs[batch_idx[full_mask], max_len - 1] = labels[full_mask]

        # Mask items for training
        masked_seqs, mask_labels = self._mask_sequence(item_seqs)
        all_item_emb = (
            self.get_all_embeddings(device=masked_seqs.device)
            if self._can_reuse_all_item_embeddings()
            else None
        )
        hidden = self.get_representation(masked_seqs, all_item_embeddings=all_item_emb)

        # Compute loss only at masked positions
        masked_positions = (mask_labels != 0)
        masked_hidden = hidden[masked_positions]  # [n_masked, hidden_size]
        targets = mask_labels[masked_positions]

        if masked_hidden.size(0) == 0:
            loss = torch.tensor(0.0, device=hidden.device, requires_grad=True)
        else:
            test_item_emb = all_item_emb if all_item_emb is not None else self.get_all_embeddings(hidden.device)
            logits = torch.matmul(masked_hidden, test_item_emb.transpose(0, 1))

            if self.config['loss_type'] == 'ce':
                loss = self.loss_func(logits, targets)
            elif self.config['loss_type'] == 'bce':
                pos_scores = torch.gather(logits, 1, targets.view(-1, 1))
                neg_ids = torch.randint(
                    self.config['select_pool'][0], self.config['select_pool'][1],
                    (targets.size(0), 1), device=targets.device
                )
                neg_scores = torch.gather(logits, 1, neg_ids)
                scores = torch.cat([pos_scores, neg_scores], dim=1).view(-1, 1)
                labels = torch.cat([
                    torch.ones_like(pos_scores),
                    torch.zeros_like(neg_scores)
                ], dim=1).view(-1, 1)
                loss = self.loss_func(scores, labels)

        # MoE auxiliary loss
        if str(self.config.get('item_adapter_type', 'linear')).lower() == 'moe':
            moe_balance_weight = float(self.config.get('moe_balance_weight', 0.0))
            if moe_balance_weight > 0:
                aux_loss = self.item_embeddings_adapter.get_aux_loss(reset=True)
                if aux_loss is not None:
                    loss = loss + moe_balance_weight * aux_loss

        return {'loss': loss}

    def predict(self, batch, n_return_sequences=1):
        """Inference: append [MASK] after the last valid item and predict next item."""
        item_seqs = batch['item_seqs'].clone()
        seq_lengths = batch['seq_lengths']
        max_len = self.config['max_seq_length']

        if isinstance(seq_lengths, torch.Tensor):
            mask_pos = seq_lengths.clone()
        else:
            mask_pos = torch.full((item_seqs.size(0),), seq_lengths,
                                  dtype=torch.long, device=item_seqs.device)

        # If sequence already fills max_seq_length, shift left by 1 to make room
        full_mask = mask_pos >= max_len
        if full_mask.any():
            item_seqs[full_mask] = torch.roll(item_seqs[full_mask], -1, dims=1)
            mask_pos[full_mask] = max_len - 1

        batch_idx = torch.arange(item_seqs.size(0), device=item_seqs.device)
        item_seqs[batch_idx, mask_pos] = self.mask_token_id

        all_item_emb = (
            self.get_all_embeddings(device=item_seqs.device)
            if self._can_reuse_all_item_embeddings()
            else None
        )
        hidden = self.get_representation(item_seqs, all_item_embeddings=all_item_emb)
        state_hidden = gather_indexes(hidden, mask_pos)

        test_item_emb = all_item_emb if all_item_emb is not None else self.get_all_embeddings(state_hidden.device)
        scores = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))[
                 :, self.config['select_pool'][0]: self.config['select_pool'][1]]
        preds = scores.topk(n_return_sequences, dim=-1).indices + self.config['select_pool'][0]
        return preds
