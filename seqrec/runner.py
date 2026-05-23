import torch
import numpy as np
from typing import Union

from accelerate import Accelerator
from torch.utils.data import DataLoader, Sampler

from .recdata import NormalRecData
from .base import AbstractModel

from .utils import get_config, init_device, init_seed, get_model, get_file_name, diagonalize_and_scale
from .trainer import BaseTrainer


class Runner:
    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    @staticmethod
    def _unwrap_compiled_model(model):
        unwrapped = model
        while hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
        return unwrapped

    def _init_accelerator(self):
        self.use_wandb = self._as_bool(self.config.get('use_wandb', False))
        if not self.use_wandb:
            return Accelerator()
        return Accelerator(log_with='wandb')

    def __init__(
            self,
            model_name: Union[str, AbstractModel],
            config_dict: dict = None,
            config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            config_file=config_file,
            config_dict=config_dict
        )
        print(self.config)

        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device()

        self.accelerator = self._init_accelerator()

        self.config['accelerator'] = self.accelerator
        self.config['use_wandb'] = self.use_wandb

        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        _ = NormalRecData(self.config).load_data()

        self.recdata = {
            'train': _[0],
            'valid': _[1],
            'test': _[2]
        }
        self.config['select_pool'] = _[3]
        self.config['item_num'] = _[4]
        self.config['eos_token'] = _[4] + 1

        if self.config['embedding']:
            pretrained_item_embeddings = torch.tensor(np.load(self.config['embedding']), dtype=torch.float32).to(self.config['device'])
            # judge if "seq_embedding" in config.keys()
            if "seq_embedding" in self.config.keys() and self.config['seq_embedding']:
                base_seq_embedding_path = self.config['seq_embedding']
                train_seq_embedding_path = base_seq_embedding_path.format("train")
                valid_seq_embedding_path = base_seq_embedding_path.format("val")
                test_seq_embedding_path = base_seq_embedding_path.format("test")
                train_seq_embedding = torch.tensor(np.load(train_seq_embedding_path), dtype=torch.float32).to(self.config['device'])
                valid_seq_embedding = torch.tensor(np.load(valid_seq_embedding_path), dtype=torch.float32).to(self.config['device'])
                test_seq_embedding = torch.tensor(np.load(test_seq_embedding_path), dtype=torch.float32).to(self.config['device'])
                pretrained_item_embeddings = [pretrained_item_embeddings, train_seq_embedding, valid_seq_embedding, test_seq_embedding]

        else:
            pretrained_item_embeddings = None

        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, pretrained_item_embeddings)
        self.model = self._maybe_compile_model(self.model)

        print(self.model)
        self.trainer = BaseTrainer(self.config, self.model)

    def _maybe_compile_model(self, model):
        if not bool(self.config.get('torch_compile', True)):
            return model
        if not hasattr(torch, 'compile'):
            print('[seqrec] torch.compile is unavailable; fallback to eager mode.')
            return model
        backend = str(self.config.get('torch_compile_backend', 'inductor'))
        mode = str(self.config.get('torch_compile_mode', 'reduce-overhead'))
        try:
            compiled_model = torch.compile(model, backend=backend, mode=mode)
            print(f'[seqrec] torch.compile enabled: backend={backend} mode={mode}')
            return compiled_model
        except Exception as exc:
            print(f'[seqrec] torch.compile failed, fallback to eager mode: {exc}')
            return model

    def run(self):
        num_workers = self.config.get('num_workers', 4)
        collate_fn = getattr(self.model, 'get_collate_fn', lambda: None)()
        train_dataloader = DataLoader(
                self.recdata['train'],
                batch_size=self.config['train_batch_size'],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        val_dataloader = DataLoader(
            self.recdata['valid'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        test_dataloader = DataLoader(
            self.recdata['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # skip training for ItemKNN model
        if self.config['model'] != 'ItemKNN':
            self.trainer.train(train_dataloader, val_dataloader)

            self.accelerator.wait_for_everyone()
            self.model = self.accelerator.unwrap_model(self.model)
            self.model = self._unwrap_compiled_model(self.model)

            if self.config.get('steps', None) != 0:
                self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt, weights_only=True))

            self.model = self._maybe_compile_model(self.model)
            self.model, test_dataloader = self.accelerator.prepare(
                self.model, test_dataloader
            )
            if self.accelerator.is_main_process:
                print(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        if self.config.get('steps', None) != 0:
            test_results = self.trainer.evaluate(test_dataloader)
            print(test_results)
            if self.use_wandb and self.accelerator.is_main_process:
                for key in test_results:
                    self.accelerator.log({f'Test_Metric/{key}': test_results[key]})

        if self.accelerator.is_main_process:
            if self.config['save'] is False:
                import os
                if os.path.exists(self.trainer.saved_model_ckpt):
                    os.remove(self.trainer.saved_model_ckpt)
                    print(f"{self.trainer.saved_model_ckpt} has been deleted.")
                else:
                    print(f"{self.trainer.saved_model_ckpt} not found.")

        self.trainer.end()
        return test_results, self.config

