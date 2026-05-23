import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import default_collate
from seqrec.base import AbstractModel
from seqrec.modules import MoEAdaptorLayer
from ..Embedding2 import Embedding2


def build_session_graphs(item_seqs, seq_lengths):
    """Build session graphs from a batch of item sequences.

    Runs on CPU in DataLoader workers for parallelism.

    Args:
        item_seqs: [batch, max_seq_length] padded item sequences (tensor)
        seq_lengths: [batch] actual sequence lengths (tensor or int)

    Returns:
        alias: [batch, max_seq_length] mapping from seq positions to node indices
        A: [batch, max_nodes, 2*max_nodes] concatenated [A_in | A_out]
        items: [batch, max_nodes] unique item ids per session
        mask: [batch, max_nodes] node validity mask
    """
    batch_size = item_seqs.size(0)
    max_seq_len = item_seqs.size(1)

    if isinstance(seq_lengths, int):
        seq_lengths_list = [seq_lengths] * batch_size
    else:
        seq_lengths_list = seq_lengths.tolist()

    item_seqs_np = item_seqs.numpy()

    all_unique_items = []
    all_alias = []
    all_adj_in = []
    all_adj_out = []
    max_n_nodes = 0

    for i in range(batch_size):
        seq_len = seq_lengths_list[i]
        seq = item_seqs_np[i, :seq_len]

        # Unique items preserving order
        unique_items = list(dict.fromkeys(seq.tolist()))
        n_nodes = len(unique_items)
        max_n_nodes = max(max_n_nodes, n_nodes)

        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

        # Build binary adjacency to match the official SR-GNN implementations.
        adj_in = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        adj_out = np.zeros((n_nodes, n_nodes), dtype=np.float32)

        if seq_len > 1:
            src_ids = np.array([item_to_idx[s] for s in seq[:-1].tolist()])
            dst_ids = np.array([item_to_idx[s] for s in seq[1:].tolist()])
            adj_out[src_ids, dst_ids] = 1.0
            adj_in[dst_ids, src_ids] = 1.0

        # Row-normalize
        row_sum_in = adj_in.sum(axis=1, keepdims=True)
        row_sum_in[row_sum_in == 0] = 1
        adj_in /= row_sum_in

        row_sum_out = adj_out.sum(axis=1, keepdims=True)
        row_sum_out[row_sum_out == 0] = 1
        adj_out /= row_sum_out

        alias = [item_to_idx[item] for item in seq.tolist()]

        all_unique_items.append(unique_items)
        all_alias.append(alias)
        all_adj_in.append(adj_in)
        all_adj_out.append(adj_out)

    # Pad to max_n_nodes and assemble tensors (on CPU — DataLoader will handle pin_memory)
    items_padded = torch.zeros(batch_size, max_n_nodes, dtype=torch.long)
    alias_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    A_padded = torch.zeros(batch_size, max_n_nodes, 2 * max_n_nodes, dtype=torch.float32)
    node_mask = torch.zeros(batch_size, max_n_nodes, dtype=torch.float32)

    for i in range(batch_size):
        n_nodes = len(all_unique_items[i])
        items_padded[i, :n_nodes] = torch.as_tensor(all_unique_items[i], dtype=torch.long)
        seq_len = seq_lengths_list[i]
        alias_padded[i, :seq_len] = torch.as_tensor(all_alias[i], dtype=torch.long)
        A_padded[i, :n_nodes, :n_nodes] = torch.from_numpy(all_adj_in[i])
        A_padded[i, :n_nodes, max_n_nodes:max_n_nodes + n_nodes] = torch.from_numpy(all_adj_out[i])
        node_mask[i, :n_nodes] = 1.0

    return alias_padded, A_padded, items_padded, node_mask


def srgnn_collate_fn(batch):
    """Custom collate that pre-builds session graphs in DataLoader workers."""
    collated = default_collate(batch)
    alias, A, graph_items, node_mask = build_session_graphs(
        collated['item_seqs'], collated['seq_lengths']
    )
    collated['alias'] = alias
    collated['A'] = A
    collated['graph_items'] = graph_items
    collated['node_mask'] = node_mask
    return collated


class GNN(nn.Module):
    """Gated Graph Neural Network for session graph."""

    def __init__(self, hidden_size, n_steps=1):
        super(GNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        # Separate linear transforms for incoming and outgoing edge messages
        self.linear_edge_in = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(hidden_size, hidden_size, bias=True)
        # Extra biases applied after adjacency multiplication (following official SR-GNN)
        self.b_iah = nn.Parameter(torch.zeros(hidden_size))
        self.b_oah = nn.Parameter(torch.zeros(hidden_size))
        # GRU cell: input is concatenated [in_msg, out_msg] of size 2D
        self.gru_cell = nn.GRUCell(2 * hidden_size, hidden_size)

    def forward(self, A, node_hidden):
        """
        Args:
            A: adjacency matrix [batch, n_nodes, 2*n_nodes] (concatenated [A_in | A_out])
            node_hidden: node embeddings [batch, n_nodes, hidden_size]
        Returns:
            Updated node_hidden [batch, n_nodes, hidden_size]
        """
        for _ in range(self.n_steps):
            B, N, D = node_hidden.shape
            # Split adjacency into incoming and outgoing
            A_in = A[:, :, :N]    # [B, N, N]
            A_out = A[:, :, N:]   # [B, N, N]
            # Aggregate messages from incoming and outgoing neighbors separately
            input_in = torch.bmm(A_in, self.linear_edge_in(node_hidden)) + self.b_iah    # [B, N, D]
            input_out = torch.bmm(A_out, self.linear_edge_out(node_hidden)) + self.b_oah # [B, N, D]
            # Concatenate to preserve directional information
            inputs = torch.cat([input_in, input_out], dim=-1)  # [B, N, 2D]
            # GRU update
            node_hidden = self.gru_cell(
                inputs.view(B * N, 2 * D),
                node_hidden.view(B * N, D)
            ).view(B, N, D)
        return node_hidden


class SRGNN(AbstractModel):
    """SR-GNN: Session-based Recommendation with Graph Neural Networks.

    Constructs a directed session graph from the item sequence, applies GGNN for
    message passing, and uses attention readout to produce the session representation.
    """

    def __init__(self, config: dict, pretrained_item_embeddings=None):
        super(SRGNN, self).__init__(config=config)
        self.config = config
        self.hidden_size = config['hidden_size']
        self.n_gnn_steps = config.get('n_gnn_steps', 1)
        self.uses_pretrained_item_embeddings = pretrained_item_embeddings is not None

        self.load_item_embeddings(pretrained_item_embeddings)

        # Graph Neural Network
        self.gnn = GNN(self.hidden_size, self.n_gnn_steps)

        # Attention readout
        self.attn_linear_q = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.attn_linear_k = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.attn_v = nn.Linear(self.hidden_size, 1, bias=False)

        # Session representation
        self.session_linear = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)

        # Dropout (applied to node embeddings before GNN)
        self.dropout = nn.Dropout(config.get('dropout', 0.0))

        # Loss
        if config['loss_type'] == 'bce':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif config['loss_type'] == 'ce':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {config['loss_type']}")

        self._reset_parameters()

    def _reset_parameters(self):
        if not self.uses_pretrained_item_embeddings:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for param in self.parameters():
                if param.requires_grad:
                    nn.init.uniform_(param, -stdv, stdv)
            return

        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Skip embedding weights — preserve init from load_item_embeddings()
                if 'pretrained_item_embeddings' in name or 'item_embeddings' in name:
                    continue
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def load_item_embeddings(self, pretrained_embs):
        if pretrained_embs is None:
            self.item_embeddings = nn.Embedding(
                num_embeddings=self.config['item_num'] + 1,
                embedding_dim=self.config['hidden_size'],
                padding_idx=0
            )
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
        return self.item_embeddings(items)

    def _compute_all_embeddings(self):
        if isinstance(self.item_embeddings, Embedding2):
            # Embedding2.weight returns a Weight dataclass; .data is the adapter output (differentiable)
            return self.item_embeddings.weight.data
        # Return Parameter directly to maintain gradient flow (not .data which detaches)
        return self.item_embeddings.weight

    def get_all_embeddings(self, device=None):
        return self._get_cached_all_embeddings(self._compute_all_embeddings, device=device)

    def _can_reuse_all_item_embeddings(self):
        return (
            hasattr(self.item_embeddings, 'can_reuse_all_embeddings')
            and self.item_embeddings.can_reuse_all_embeddings()
        )

    @staticmethod
    def get_collate_fn():
        """Return custom collate function for DataLoader that pre-builds session graphs."""
        return srgnn_collate_fn

    def _attention_readout(self, node_hidden, alias, seq_lengths):
        """Soft attention readout over sequence positions (not unique nodes).

        Following the official SR-GNN: after GNN propagation on unique nodes,
        re-map node embeddings back to sequence positions using alias, then
        compute attention over the full sequence.

        Args:
            node_hidden: [batch, max_nodes, hidden_size] GNN output on unique nodes
            alias: [batch, max_seq_length] mapping from seq positions to node indices
            seq_lengths: [batch] or scalar

        Returns:
            session_repr: [batch, hidden_size]
        """
        batch_size = alias.size(0)
        max_seq_len = alias.size(1)

        # Re-map node embeddings to sequence positions
        alias_expanded = alias.unsqueeze(-1).expand(-1, -1, self.hidden_size)  # [B, seq_len, D]
        seq_hidden = node_hidden.gather(1, alias_expanded)  # [B, seq_len, D]

        # Build sequence mask from seq_lengths
        if isinstance(seq_lengths, torch.Tensor):
            last_idx = seq_lengths - 1
            seq_mask = torch.arange(max_seq_len, device=alias.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        else:
            last_idx = torch.full((batch_size,), seq_lengths - 1,
                                  dtype=torch.long, device=alias.device)
            seq_mask = torch.arange(max_seq_len, device=alias.device).unsqueeze(0) < seq_lengths
        seq_mask = seq_mask.float()  # [B, seq_len]

        # Local embedding: last clicked item in sequence
        local_emb = seq_hidden.gather(
            1, last_idx.view(-1, 1, 1).expand(-1, -1, self.hidden_size)
        ).squeeze(1)  # [B, D]

        # Attention over sequence positions
        q = self.attn_linear_q(local_emb).unsqueeze(1)  # [B, 1, D]
        k = self.attn_linear_k(seq_hidden)  # [B, seq_len, D]
        attn_scores = self.attn_v(torch.sigmoid(q + k)).squeeze(-1)  # [B, seq_len]

        # Mask padding positions and compute weighted sum
        attn_weights = attn_scores * seq_mask  # zero out padding
        global_emb = torch.sum(attn_weights.unsqueeze(-1) * seq_hidden * seq_mask.unsqueeze(-1), dim=1)  # [B, D]

        # Session representation
        session_repr = self.session_linear(torch.cat([local_emb, global_emb], dim=-1))  # [B, D]
        return session_repr

    def get_session_representation(self, batch, all_item_embeddings=None):
        """Run GNN on pre-built graphs and produce session representation."""
        alias = batch['alias']
        A = batch['A']
        unique_items = batch['graph_items']
        seq_lengths = batch['seq_lengths']

        # Get node embeddings and apply dropout
        if all_item_embeddings is not None:
            node_hidden = all_item_embeddings[unique_items]
        else:
            node_hidden = self.get_embeddings(unique_items)
        node_hidden = self.dropout(node_hidden)  # [batch, max_nodes, hidden_size]

        # GNN propagation
        node_hidden = self.gnn(A, node_hidden)

        # Attention readout over sequence positions
        session_repr = self._attention_readout(node_hidden, alias, seq_lengths)
        return session_repr

    def forward(self, batch):
        all_item_emb = (
            self.get_all_embeddings(device=batch['graph_items'].device)
            if self._can_reuse_all_item_embeddings()
            else None
        )
        state_hidden = self.get_session_representation(batch, all_item_embeddings=all_item_emb)
        test_item_emb = all_item_emb if all_item_emb is not None else self.get_all_embeddings(state_hidden.device)
        candidate_start, candidate_end = self.config['select_pool']

        if self.config['loss_type'] == 'bce':
            logits = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))
            pos_scores = torch.gather(logits, 1, batch['labels'].view(-1, 1))
            # Random negative sampling
            target_neg = []
            for index in range(len(batch['labels'])):
                neg = np.random.randint(self.config['select_pool'][0], self.config['select_pool'][1])
                while neg == batch['labels'][index]:
                    neg = np.random.randint(self.config['select_pool'][0], self.config['select_pool'][1])
                target_neg.append(neg)
            labels_neg = torch.LongTensor(target_neg).to(batch['labels'].device).reshape(-1, 1)
            neg_scores = torch.gather(logits, 1, labels_neg)
            pos_labels = torch.ones((batch['labels'].shape[0], 1), device=state_hidden.device)
            neg_labels = torch.zeros((batch['labels'].shape[0], 1), device=state_hidden.device)
            scores = torch.cat((pos_scores, neg_scores), dim=1).view(-1, 1)
            labels = torch.cat((pos_labels, neg_labels), dim=1).view(-1, 1)
            loss = self.loss_func(scores, labels)

        elif self.config['loss_type'] == 'ce':
            candidate_emb = test_item_emb[candidate_start:candidate_end]
            logits = torch.matmul(state_hidden, candidate_emb.transpose(0, 1))
            loss = self.loss_func(logits, batch['labels'].view(-1) - candidate_start)

        # MoE auxiliary loss
        if str(self.config.get('item_adapter_type', 'linear')).lower() == 'moe':
            moe_balance_weight = float(self.config.get('moe_balance_weight', 0.0))
            if moe_balance_weight > 0:
                aux_loss = self.item_embeddings_adapter.get_aux_loss(reset=True)
                if aux_loss is not None:
                    loss = loss + moe_balance_weight * aux_loss

        return {'loss': loss}

    def predict(self, batch, n_return_sequences=1):
        all_item_emb = (
            self.get_all_embeddings(device=batch['graph_items'].device)
            if self._can_reuse_all_item_embeddings()
            else None
        )
        state_hidden = self.get_session_representation(batch, all_item_embeddings=all_item_emb).view(
            -1, self.hidden_size
        )
        test_item_emb = all_item_emb if all_item_emb is not None else self.get_all_embeddings(state_hidden.device)
        scores = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))[
                 :, self.config['select_pool'][0]: self.config['select_pool'][1]]
        preds = scores.topk(n_return_sequences, dim=-1).indices + self.config['select_pool'][0]
        return preds
