from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


DEFAULT_COOCCURRENCE_MAX_EXACT_COUNT = 6


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_cooccurrence_count_mode(mode: str) -> str:
    aliases = {
        "adjacent": "adjacent",
        "transition": "adjacent",
        "next_item": "adjacent",
        "next-item": "adjacent",
        "sliding_window": "sliding_window",
        "sliding-window": "sliding_window",
        "window": "sliding_window",
        "cooccurrence": "sliding_window",
        "co-occurrence": "sliding_window",
        "global": "global",
        "session": "global",
        "sequence": "global",
    }
    key = str(mode or "sliding_window").strip().lower()
    if key not in aliases:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported cooccurrence_count_mode: {mode}. Supported: {supported}")
    return aliases[key]


def _clean_sequence(sequence: Sequence[int]) -> List[int]:
    return [int(item_id) for item_id in sequence if int(item_id) > 0]


def _pair_key(source_item_id: int, target_item_id: int, symmetric: bool) -> Tuple[int, int]:
    source_item_id = int(source_item_id)
    target_item_id = int(target_item_id)
    if symmetric and target_item_id < source_item_id:
        return target_item_id, source_item_id
    return source_item_id, target_item_id


def _iter_sliding_windows(sequence: Sequence[int], window_size: int) -> Iterable[List[int]]:
    effective_window_size = int(window_size)
    if effective_window_size < 2:
        raise ValueError(f"cooccurrence_window_size must be >= 2, got {window_size}")
    cleaned_sequence = _clean_sequence(sequence)
    if len(cleaned_sequence) < 2:
        return
    if len(cleaned_sequence) <= effective_window_size:
        yield cleaned_sequence
        return
    for start in range(len(cleaned_sequence) - effective_window_size + 1):
        yield cleaned_sequence[start:start + effective_window_size]


def _unique_items_for_pair_count(item_ids: Sequence[int], symmetric: bool) -> List[int]:
    cleaned = _clean_sequence(item_ids)
    if symmetric:
        return sorted(set(cleaned))

    unique_items = []
    seen = set()
    for item_id in cleaned:
        if item_id in seen:
            continue
        unique_items.append(item_id)
        seen.add(item_id)
    return unique_items


def _update_cooccurrence_pair_counts(
    item_ids: Sequence[int],
    pair_counts: Dict[Tuple[int, int], int],
    symmetric: bool,
) -> None:
    unique_items = _unique_items_for_pair_count(item_ids, symmetric=symmetric)
    for source_idx in range(len(unique_items)):
        source_item_id = int(unique_items[source_idx])
        for target_idx in range(source_idx + 1, len(unique_items)):
            target_item_id = int(unique_items[target_idx])
            if target_item_id <= 0 or target_item_id == source_item_id:
                continue
            key = _pair_key(source_item_id, target_item_id, symmetric=symmetric)
            pair_counts[key] += 1


def build_train_cooccurrence_counts(
    train_sequences: Sequence[Sequence[int]],
    mode: str = "sliding_window",
    window_size: int = 3,
    symmetric: Optional[bool] = None,
) -> Dict[Tuple[int, int], int]:
    """Count training-set item-pair co-occurrence for evaluation bucketing.

    The default mirrors the Stage 2 mainline pair mining setup: sliding-window
    item co-occurrence with window_size=3 and symmetric item pairs.
    """
    canonical_mode = _normalize_cooccurrence_count_mode(mode)
    effective_symmetric = canonical_mode != "adjacent" if symmetric is None else _as_bool(symmetric)
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    for sequence in train_sequences:
        cleaned_sequence = _clean_sequence(sequence)
        if len(cleaned_sequence) < 2:
            continue

        if canonical_mode == "adjacent":
            for source_item_id, target_item_id in zip(cleaned_sequence[:-1], cleaned_sequence[1:]):
                if source_item_id == target_item_id:
                    continue
                key = _pair_key(source_item_id, target_item_id, symmetric=effective_symmetric)
                pair_counts[key] += 1
            continue

        if canonical_mode == "global":
            _update_cooccurrence_pair_counts(cleaned_sequence, pair_counts, symmetric=effective_symmetric)
            continue

        for window_item_ids in _iter_sliding_windows(cleaned_sequence, window_size=int(window_size)):
            _update_cooccurrence_pair_counts(window_item_ids, pair_counts, symmetric=effective_symmetric)

    return dict(pair_counts)


def make_cooccurrence_bucket_labels(max_exact_count: int = DEFAULT_COOCCURRENCE_MAX_EXACT_COUNT) -> List[str]:
    effective_max = int(max_exact_count)
    return [str(count) for count in range(effective_max + 1)] + [f"gt{effective_max}"]


def _cooccurrence_count_to_bucket(count: int, max_exact_count: int) -> int:
    count = int(count)
    effective_max = int(max_exact_count)
    if count <= effective_max:
        return max(0, count)
    return effective_max + 1


def build_eval_sample_cooccurrence_counts(
    sequences: Sequence[Sequence[int]],
    pair_counts: Dict[Tuple[int, int], int],
    symmetric: bool,
) -> List[int]:
    counts: List[int] = []
    for sequence in sequences:
        cleaned_sequence = _clean_sequence(sequence)
        if len(cleaned_sequence) < 2:
            counts.append(0)
            continue
        source_item_id = int(cleaned_sequence[-2])
        target_item_id = int(cleaned_sequence[-1])
        key = _pair_key(source_item_id, target_item_id, symmetric=symmetric)
        counts.append(int(pair_counts.get(key, 0)))
    return counts


class SequenceDataset(Dataset):
    def __init__(
        self,
        config,
        sequences,
        seq_type=None,
        cooccurrence_counts: Optional[Sequence[int]] = None,
        cooccurrence_max_exact_count: int = DEFAULT_COOCCURRENCE_MAX_EXACT_COUNT,
    ):
        self.sequences = sequences
        self.config = config
        self.seq_type = seq_type
        self.cooccurrence_counts = None if cooccurrence_counts is None else [int(x) for x in cooccurrence_counts]
        self.cooccurrence_max_exact_count = int(cooccurrence_max_exact_count)
        if self.cooccurrence_counts is not None and len(self.cooccurrence_counts) != len(self.sequences):
            raise ValueError("cooccurrence_counts must have the same length as sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        item_seq = seq[:-1]
        labels = seq[-1]
        seq_length = len(item_seq)
        padding_length = self.config['max_seq_length'] - len(item_seq)
        if padding_length > 0:
            item_seq = item_seq + [0] * padding_length
        sample = {
            'item_seqs': torch.tensor(item_seq, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'seq_lengths': seq_length,

            # The variables below are used for sequential embedding generation. Ignore if not needed.
            'seq_ids': idx,
            'seq_type': self.seq_type
        }
        if self.cooccurrence_counts is not None:
            cooccurrence_count = int(self.cooccurrence_counts[idx])
            sample['cooccurrence_counts'] = torch.tensor(cooccurrence_count, dtype=torch.long)
            sample['cooccurrence_buckets'] = torch.tensor(
                _cooccurrence_count_to_bucket(cooccurrence_count, self.cooccurrence_max_exact_count),
                dtype=torch.long,
            )
        return sample


class NormalRecData:
    def __init__(self, config: dict):
        self.config = config

    def load_data(self):
        from pathlib import Path

        source_dict = {
            "Games_5core": "Video_Games/5-core/downstream",
            "Arts_5core": "Arts_Crafts_and_Sewing/5-core/downstream",
            "Baby_5core": "Baby_Products/5-core/downstream",
        }
        self.config['source_dict'] = source_dict

        def read_data_from_file(domain, mode=''):
            base_path = Path('data/')
            file_path = base_path / source_dict[domain] / '{}data.txt'.format(mode)
            with file_path.open('r') as file:
                item_seqs = [list(map(int, line.split()))[-self.config['max_seq_length']-1:] for line in file]

            if mode == '':
                flat_list = [item for sublist in item_seqs for item in sublist]
                import numpy as np
                item_num = np.max(flat_list)
                return item_seqs, item_num
            else:
                return item_seqs

        train_data = []
        valid_data = []
        test_data = []

        tmp_item_seqs, total_item_num = read_data_from_file(self.config['dataset'])
        tmp_train_item_seqs, tmp_valid_item_seqs, tmp_test_item_seqs = (
            read_data_from_file(self.config['dataset'], mode='train_'),
            read_data_from_file(self.config['dataset'], mode='val_'),
            read_data_from_file(self.config['dataset'], mode='test_')
            )
        train_data.extend(tmp_train_item_seqs)
        valid_data.extend(tmp_valid_item_seqs)
        test_data.extend(tmp_test_item_seqs)
        select_pool = [1, total_item_num + 1]

        cooccurrence_bucket_eval = _as_bool(
            self.config.get('cooccurrence_bucket_eval', self.config.get('eval_by_cooccurrence', False))
        )
        valid_cooccurrence_counts = None
        test_cooccurrence_counts = None
        cooccurrence_max_exact_count = int(
            self.config.get('cooccurrence_max_exact_count', DEFAULT_COOCCURRENCE_MAX_EXACT_COUNT)
        )
        if cooccurrence_bucket_eval:
            count_mode = _normalize_cooccurrence_count_mode(
                self.config.get('cooccurrence_count_mode', self.config.get('cooccurrence_eval_mode', 'sliding_window'))
            )
            window_size = int(
                self.config.get('cooccurrence_window_size', self.config.get('cooccurrence_eval_window_size', 3))
            )
            symmetric_value = self.config.get('cooccurrence_symmetric', None)
            symmetric = count_mode != 'adjacent' if symmetric_value is None else _as_bool(symmetric_value)
            pair_counts = build_train_cooccurrence_counts(
                train_data,
                mode=count_mode,
                window_size=window_size,
                symmetric=symmetric,
            )
            valid_cooccurrence_counts = build_eval_sample_cooccurrence_counts(
                valid_data,
                pair_counts=pair_counts,
                symmetric=symmetric,
            )
            test_cooccurrence_counts = build_eval_sample_cooccurrence_counts(
                test_data,
                pair_counts=pair_counts,
                symmetric=symmetric,
            )
            self.config['cooccurrence_bucket_eval'] = True
            self.config['cooccurrence_count_mode'] = count_mode
            self.config['cooccurrence_window_size'] = window_size
            self.config['cooccurrence_symmetric'] = symmetric
            self.config['cooccurrence_max_exact_count'] = cooccurrence_max_exact_count
            self.config['cooccurrence_bucket_labels'] = make_cooccurrence_bucket_labels(cooccurrence_max_exact_count)
            self.config['cooccurrence_train_unique_pair_count'] = int(len(pair_counts))
            self.config['cooccurrence_train_raw_pair_count'] = int(sum(pair_counts.values()))

        return (
            SequenceDataset(self.config, train_data, seq_type='train'),
            SequenceDataset(
                self.config,
                valid_data,
                seq_type='val',
                cooccurrence_counts=valid_cooccurrence_counts,
                cooccurrence_max_exact_count=cooccurrence_max_exact_count,
            ),
            SequenceDataset(
                self.config,
                test_data,
                seq_type='test',
                cooccurrence_counts=test_cooccurrence_counts,
                cooccurrence_max_exact_count=cooccurrence_max_exact_count,
            ),
            select_pool,
            total_item_num
        )
