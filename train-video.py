from tqdm import tqdm
import logging
import argparse
import math
import torch
import torchvision as tv
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.utils import unwrap_model

from lvae.trainer import BaseTrainingWrapper
from lvae.datasets.video import Vimeo90k


def parse_args():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    # wandb setting
    parser.add_argument('--wbproject',  type=str,  default='qres-video')
    parser.add_argument('--wbgroup',    type=str,  default='default')
    parser.add_argument('--wbtags',     type=str,  default=None, nargs='+')
    parser.add_argument('--wbnote',     type=str,  default=None)
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    parser.add_argument('--name',       type=str,  default=None)
    # model setting
    parser.add_argument('--model',      type=str,  default='cspy')
    parser.add_argument('--model_args', type=str,  default='')
    # resume setting
    parser.add_argument('--resume',     type=str,  default=None)
    parser.add_argument('--weights',    type=str,  default=None)
    parser.add_argument('--load_optim', action=argparse.BooleanOptionalAction, default=False)
    # data setting
    # parser.add_argument('--trainset',   type=str,  default='-')
    parser.add_argument('--tr_frames',  type=int,  default=3)
    parser.add_argument('--valset',     type=str,  default='uvg-1080p')
    parser.add_argument('--val_frames', type=int,  default=12)
    # optimization setting
    parser.add_argument('--batch_size', type=int,  default=4)
    parser.add_argument('--accum_num',  type=int,  default=1)
    parser.add_argument('--optimizer',  type=str,  default='adam')
    parser.add_argument('--lr',         type=float,default=2e-4)
    parser.add_argument('--lr_sched',   type=str,  default='constant')
    parser.add_argument('--lr_warmup',  type=int,  default=1000)
    # parser.add_argument('--wdecay',     type=float,default=0.0)
    parser.add_argument('--grad_clip',  type=float,default=2.0)
    # training iterations setting
    parser.add_argument('--iterations', type=int,  default=1_000_000)
    parser.add_argument('--log_itv',    type=int,  default=100)
    parser.add_argument('--study_itv',  type=int,  default=1000)
    # parser.add_argument('--save_per',   type=int,  default=1000)
    parser.add_argument('--eval_itv',   type=int,  default=2000)
    parser.add_argument('--eval_first', action=argparse.BooleanOptionalAction, default=False)
    # exponential moving averaging (EMA)
    parser.add_argument('--ema',        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ema_decay',  type=float,default=0.9999)
    parser.add_argument('--ema_warmup', type=int,  default=10_000)
    # device setting
    parser.add_argument('--fixseed',    action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--workers',    type=int,  default=0)
    cfg = parser.parse_args()

    cfg.wdecay = 0.0
    cfg.amp = False
    return cfg


def make_generator(dataset, batch_size, workers):
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=workers, pin_memory=True)
    while True:
        yield from dataloader


class TrainWrapper(BaseTrainingWrapper):
    def __init__(self, cfg):
        super().__init__()
        self.main(cfg)

    def main(self, cfg):
        self.cfg = cfg

        # preparation
        self.set_logging()
        self.set_device()
        self.prepare_configs()
        self.set_dataset()
        self.set_model()
        self.set_optimizer()
        self.set_pretrain()

        # logging
        self.ema = None
        if self.is_main:
            self.set_wandb()
            self.set_ema()

        # the main training loops
        self.training_loops()

    def prepare_configs(self):
        super().prepare_configs()
        cfg = self.cfg
        self.adjust_lr_interval = 10
        self.ddp_check_interval = cfg.eval_itv
        self.model_log_interval = cfg.study_itv
        self.wandb_log_interval = cfg.log_itv

    def set_dataset(self):
        cfg = self.cfg

        logging.info('Initializing Datasets and Dataloaders...')
        trainset = Vimeo90k(n_frames=cfg.tr_frames)
        trainloader = make_generator(trainset, batch_size=cfg.batch_size, workers=cfg.workers)
        logging.info(f'Training root: {trainset.root}')
        logging.info(f'Number of training images = {len(trainset)}')
        logging.info(f'Training transform: \n{str(trainset.transform)}')

        self._epoch_len  = len(trainset) / cfg.bs_effective
        self.trainloader = trainloader
        # self.valloader   = valloader
        self.cfg.epochs  = float(cfg.iterations / self._epoch_len)

    def training_loops(self):
        cfg = self.cfg

        if self.distributed: # DDP mode
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        model = self.model.train()

        # ======================== initialize logging ========================
        pbar = range(self._cur_iter, cfg.iterations)
        if self.is_main:
            pbar = tqdm(pbar)
            self.init_progress_table()
        # ======================== start training ========================
        for step in pbar:
            self._cur_iter  = step
            self._cur_epoch = step / self._epoch_len

            # evaluation
            if self.is_main:
                if cfg.eval_itv <= 0: # no evaluation
                    pass
                elif (step == 0) and (not cfg.eval_first): # first iteration
                    pass
                elif step % cfg.eval_itv == 0: # evaluate every {cfg.eval_itv} epochs
                    self.evaluate()
                    model.train()
                    print(self._pbar_header)

            # learning rate schedule
            if step % self.adjust_lr_interval == 0:
                self.adjust_lr(step, cfg.iterations)

            # training step
            assert model.training
            batch = next(self.trainloader)
            stats = model(batch)
            loss = stats['loss'] / float(cfg.accum_num)
            loss.backward() # gradients are averaged over devices in DDP mode
            # parameter update
            if step % cfg.accum_num == 0:
                grad_norm, bad = self.gradient_clip(model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (self.ema is not None) and (not bad):
                    _warmup = cfg.ema_warmup or (cfg.iterations // 20)
                    self.ema.decay = cfg.ema_decay * (1 - math.exp(-step / _warmup))
                    self.ema.update(model)

            # sanity check
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.error(f'loss = {loss}')
                self.clean_and_exit()

            # logging
            if self.is_main:
                self.minibatch_log(pbar, stats)
                self.periodic_log(batch)

        self._cur_iter += 1
        if self.is_main:
            self.evaluate()
            logging.info(f'Training finished. results: \n {self._results}')

    def periodic_log(self, batch):
        assert self.is_main
        # model logging
        if self._cur_iter % self.model_log_interval == 0:
            self.model.eval()
            model = unwrap_model(self.model)
            if hasattr(model, 'study'):
                model.study(log_dir=self._log_dir/'study', wandb_run=self.wbrun)
                # self.ema.module.study(log_dir=self._log_dir/'ema')
            self.model.train()

        # Weights & Biases logging
        if self._cur_iter % self.wandb_log_interval == 0:
            imgs = batch if torch.is_tensor(batch) else batch[0]
            assert torch.is_tensor(imgs)
            N = min(16, imgs.shape[0])
            tv.utils.save_image(imgs[:N], fp=self._log_dir / 'inputs.png', nrow=math.ceil(N**0.5))

            _log_dic = {
                'general/lr': self.optimizer.param_groups[0]['lr'],
                # 'general/grad_norm': self._moving_max_grad_norm,
                'general/grad_norm': self._moving_grad_norm_buffer.max(),
                'ema/decay': (self.ema.decay if self.ema else 0)
            }
            _log_dic.update(
                {'train/'+k: self.stats_table[k] for k in self.wandb_log_keys}
            )
            self.wbrun.log(_log_dic, step=self._cur_iter)

    def eval_model(self, model):
        cfg = self.cfg
        results = model.self_evaluate(cfg.valset, max_frames=cfg.val_frames,
                                      log_dir=self._log_dir/'study')
        return results


def main():
    cfg = parse_args()
    TrainWrapper(cfg)

if __name__ == '__main__':
    main()
