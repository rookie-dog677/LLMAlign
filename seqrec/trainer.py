import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from .base import AbstractModel
from transformers.optimization import get_scheduler
from collections import defaultdict, OrderedDict
from .utils import get_file_name, get_total_steps
from .evaluator import Evaluator


class BaseTrainer(object):
    def __init__(self, config: dict, model: AbstractModel):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config)
        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='.pth')
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)
        self.best_metric = 0
        self.best_epoch = 0
        self.count = 0

        self.checkpoints_deque = []

    def _cooccurrence_bucket_eval_enabled(self) -> bool:
        return bool(self.config.get('cooccurrence_bucket_eval', False))

    def _cooccurrence_bucket_labels(self):
        labels = self.config.get('cooccurrence_bucket_labels')
        if labels:
            return [str(label) for label in labels]
        max_exact_count = int(self.config.get('cooccurrence_max_exact_count', 6))
        return [str(count) for count in range(max_exact_count + 1)] + [f'gt{max_exact_count}']

    def _build_optimizer(self):
        optimizer_name = str(self.config.get('optimizer', 'adamw')).lower()
        if optimizer_name == 'adam':
            optimizer_cls = Adam
        elif optimizer_name == 'adamw':
            optimizer_cls = AdamW
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")

        return optimizer_cls(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    def _tracking_enabled(self) -> bool:
        return self._as_bool(self.config.get('use_wandb', False))

    @staticmethod
    def _unwrap_compiled_model(model):
        unwrapped = model
        while hasattr(unwrapped, '_orig_mod'):
            unwrapped = unwrapped._orig_mod
        return unwrapped

    def train(self, train_dataloader, val_dataloader):
        optimizer = self._build_optimizer()
        scheduler = None
        scheduler_type = str(self.config.get('lr_scheduler_type', '')).lower()
        if scheduler_type in ('', 'none') and 'lr_dc' in self.config and 'lr_dc_step' in self.config:
            scheduler_type = 'step'
        if scheduler_type == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=int(self.config['lr_dc_step']),
                gamma=float(self.config['lr_dc']),
            )
        elif scheduler_type not in ('', 'none'):
            raise ValueError(f"Unsupported lr_scheduler_type: {self.config['lr_scheduler_type']}")

        total_n_steps = get_total_steps(self.config, train_dataloader)
        self.model, optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader)
        tracker_config = dict(self.config)
        tracker_config.pop('accelerator', None)
        if self._tracking_enabled():
            init_kwargs = {
                'wandb': {
                    'name': get_file_name(self.config),
                    'group': self.config.get('wandb_group', None),
                    'mode': str(self.config.get('wandb_mode', 'online')),
                }
            }
            self.accelerator.init_trackers(
                project_name=str(self.config.get('wandb_project', 'LLMAlign_Eval')),
                config=tracker_config,
                init_kwargs=init_kwargs,
            )
        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )
            for batch in train_progress_bar:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs['loss']
                self.accelerator.backward(loss)
                optimizer.step()
                total_loss = total_loss + loss.item()

            if self._tracking_enabled():
                self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)

            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                all_results = self.evaluate(val_dataloader, split='val')
                if self.accelerator.is_main_process:
                    for key in all_results:
                        if self._tracking_enabled():
                            self.accelerator.log({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                    print(all_results)

                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        if self.config['use_ddp']:  # unwrap model for saving
                            model_to_save = self.accelerator.unwrap_model(self.model)
                        else:
                            model_to_save = self.model
                        model_to_save = self._unwrap_compiled_model(model_to_save)
                        torch.save(model_to_save.state_dict(), self.saved_model_ckpt)
                        print(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')
                else:
                    print('Patience for {} Times'.format(epoch + 1 - best_epoch))

                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
            if scheduler is not None:
                scheduler.step()
        print(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')

    def evaluate(self, dataloader, split='test'):

        self.model.eval()
        predict_model = self.model.module if self.config['use_ddp'] else self.model
        cache_model = self._unwrap_compiled_model(predict_model)
        if hasattr(cache_model, 'prepare_eval_item_embeddings'):
            cache_model.prepare_eval_item_embeddings(device=self.accelerator.device)

        all_results = defaultdict(list)
        bucket_eval_enabled = self._cooccurrence_bucket_eval_enabled()
        bucket_labels = self._cooccurrence_bucket_labels() if bucket_eval_enabled else []
        bucket_results = {
            label: defaultdict(list)
            for label in bucket_labels
        }
        bucket_sample_counts = {
            label: 0
            for label in bucket_labels
        }
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) if k != "seq_type" else v for k, v in batch.items()}
                if self.config['use_ddp']:  # ddp, gather data from all devices for evaluation
                    preds = predict_model.predict(batch, n_return_sequences=self.evaluator.maxk)
                    if bucket_eval_enabled and 'cooccurrence_buckets' in batch:
                        all_preds, all_labels, all_buckets = self.accelerator.gather_for_metrics(
                            (preds, batch['labels'], batch['cooccurrence_buckets'])
                        )
                    else:
                        all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                        all_buckets = None
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds = predict_model.predict(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])
                    all_buckets = batch.get('cooccurrence_buckets')

                for key, value in results.items():
                    all_results[key].append(value)

                if bucket_eval_enabled and all_buckets is not None:
                    bucket_ids = all_buckets.detach().cpu()
                    for bucket_idx, bucket_label in enumerate(bucket_labels):
                        mask = bucket_ids == int(bucket_idx)
                        sample_count = int(mask.sum().item())
                        bucket_sample_counts[bucket_label] += sample_count
                        if sample_count == 0:
                            continue
                        for key, value in results.items():
                            bucket_results[bucket_label][key].append(value[mask])


        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()
        if bucket_eval_enabled:
            total_bucket_samples = int(sum(bucket_sample_counts.values()))
            output_results['coocc_bucket/total/count'] = total_bucket_samples
            for bucket_label in bucket_labels:
                bucket_count = int(bucket_sample_counts[bucket_label])
                output_results[f'coocc_bucket/{bucket_label}/count'] = bucket_count
                output_results[f'coocc_bucket/{bucket_label}/ratio'] = (
                    float(bucket_count) / float(total_bucket_samples)
                    if total_bucket_samples > 0 else 0.0
                )
                if bucket_count == 0:
                    continue
                for metric in self.config['metrics']:
                    for k in self.config['topk']:
                        key = f"{metric}@{k}"
                        if bucket_results[bucket_label][key]:
                            output_results[f'coocc_bucket/{bucket_label}/{key}'] = (
                                torch.cat(bucket_results[bucket_label][key]).mean().item()
                            )
        return output_results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()
