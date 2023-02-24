from tqdm import tqdm
import json
import logging
import argparse
import math
import torch
import torchvision as tv
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.utils import unwrap_model

from lvae.utils.coding import bd_rate
from lvae.paths import known_datasets
from lvae.trainer import BaseTrainingWrapper
from lvae.datasets.image import get_cls_dataset, make_generator


def parse_args():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    # wandb setting
    parser.add_argument('--wbproject',  type=str,  default='project0')
    parser.add_argument('--wbgroup',    type=str,  default='group0')
    parser.add_argument('--wbtags',     type=str,  default=None, nargs='+')
    parser.add_argument('--wbnote',     type=str,  default=None)
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    parser.add_argument('--name',       type=str,  default=None)
    # model setting
    parser.add_argument('--model',      type=str,  default='qarv_cls_test')
    parser.add_argument('--model_args', type=str,  default='')
    # resume setting
    parser.add_argument('--resume',     type=str,  default=None)
    parser.add_argument('--weights',    type=str,  default=None)
    parser.add_argument('--load_optim', action=argparse.BooleanOptionalAction, default=False)
    # data setting
    parser.add_argument('--trainset',   type=str,  default='imagenet-train')
    parser.add_argument('--transform',  type=str,  default='crop=256,hflip=True')
    parser.add_argument('--valset',     type=str,  default='kodak')
    parser.add_argument('--val_steps',  type=int,  default=8)
    # optimization setting
    parser.add_argument('--batch_size', type=int,  default=16)
    parser.add_argument('--accum_num',  type=int,  default=1)
    parser.add_argument('--optimizer',  type=str,  default='adam')
    parser.add_argument('--lr',         type=float,default=2e-4)
    parser.add_argument('--lr_sched',   type=str,  default='const-0.5-cos')
    parser.add_argument('--lrf_min',    type=float,default=0.01)
    parser.add_argument('--lr_warmup',  type=int,  default=0)
    parser.add_argument('--grad_clip',  type=float,default=2.0)
    # training iterations setting
    parser.add_argument('--iterations', type=int,  default=2_000_000)
    parser.add_argument('--log_itv',    type=int,  default=100)
    parser.add_argument('--study_itv',  type=int,  default=2000)
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

        # DDP mode
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # the main training loops
        self.training_loops()

    def prepare_configs(self):
        super().prepare_configs()
        cfg = self.cfg
        self.model_log_interval = cfg.study_itv
        self.wandb_log_interval = cfg.log_itv

    def set_dataset(self):
        cfg = self.cfg

        logging.info('Initializing Datasets and Dataloaders...')
        trainset = get_cls_dataset(cfg.trainset, transform_cfg=cfg.transform)
        trainloader = make_generator(trainset, batch_size=cfg.batch_size, workers=cfg.workers)
        logging.info(f'Training root: {trainset.root}')
        logging.info(f'Number of training images = {len(trainset)}')
        logging.info(f'Training transform: \n{str(trainset.transform)}')

        # test set
        val_img_dir = known_datasets[cfg.valset]
        logging.info(f'Val root: {val_img_dir} \n')

        self._epoch_len  = len(trainset) / cfg.bs_effective
        self.trainloader = trainloader
        # self.valloader   = valloader
        self.val_img_dir = val_img_dir
        self.cfg.epochs  = float(cfg.iterations / self._epoch_len)

    def training_loops(self):
        cfg = self.cfg
        model = self.model

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
            if step % 10 == 0:
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

                if (self.ema is not None) and not bad:
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
                model.study(save_dir=self._log_dir, wandb_run=self.wbrun)
                # self.ema.ema.study(save_dir=self._log_dir/'ema')
                self.ema.module.study(save_dir=self._log_dir/'ema')
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

    @torch.no_grad()
    def evaluate(self):
        assert self.is_main
        log_dir = self._log_dir

        # Evaluation
        _log_dic = {
            'general/epoch': self._cur_epoch,
            'general/iter':  self._cur_iter
        }
        model_ = unwrap_model(self.model).eval()
        results = model_.self_evaluate(self.val_img_dir, log_dir=log_dir, steps=self.cfg.val_steps)
        results_to_log = self.process_log_results(results)

        _log_dic.update({'val-metrics/plain-'+k: v for k,v in results_to_log.items()})
        # save last checkpoint
        checkpoint = {
            'model'     : model_.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'scaler'    : self.scaler.state_dict(),
            # loop_name   : loop_step,
            'epoch': self._cur_epoch,
            'iter':  self._cur_iter,
            'results'   : results,
        }
        torch.save(checkpoint, log_dir / 'last.pt')
        self._save_if_best(checkpoint)

        if self.cfg.ema:
            # no_ema_loss = results['loss']
            # results = self.eval_model(self.ema.module)
            results = self.ema.module.self_evaluate(self.val_img_dir, log_dir=log_dir, steps=self.cfg.val_steps)
            results_to_log = self.process_log_results(results)
            # log_json_like(results)
            _log_dic.update({'val-metrics/ema-'+k: v for k,v in results_to_log.items()})
            # save last checkpoint of EMA
            checkpoint = {
                'model': self.ema.module.state_dict(),
                'epoch': self._cur_epoch,
                'iter':  self._cur_iter,
                'results' : results,
            }
            torch.save(checkpoint, log_dir / 'last_ema.pt')
            self._save_if_best(checkpoint)

        # wandb log
        self.wbrun.log(_log_dic, step=self._cur_iter)
        # Log evaluation results to file
        msg = self.stats_table.get_body() + '||' + '%10.4g' * 1 % (results['loss'])
        with open(log_dir / 'results.txt', 'a') as f:
            f.write(msg + '\n')

        self._results = results
        print()

    def process_log_results(self, results):
        bdr = compute_bd_rate_over_anchor(results, self.cfg.valset)
        lambdas = results['lambda']
        results_to_log = {'bd-rate': bdr}
        for idx in [0, len(lambdas)//2, -1]:
            lmb = round(lambdas[idx])
            results_to_log.update({
                f'lmb{lmb}/loss': results['loss'][idx],
                f'lmb{lmb}/bpp':  results['bpp'][idx],
                f'lmb{lmb}/psnr': results['psnr'][idx],
            })
        results['loss'] = bdr
        results['bd-rate'] = bdr
        print_json_like(results)
        return results_to_log

def read_rd_stats_from_json(json_path):
    with open(json_path, mode='r') as f:
        stats = json.load(fp=f)
    assert isinstance(stats, dict)
    stats = stats.get('results', stats)
    return stats

def get_anchor_stats(dataset_name):
    anchor_paths = {
        'kodak': 'data/kodak-vtm18.0.json'
    }
    anchor_stats = read_rd_stats_from_json(anchor_paths[dataset_name])
    return anchor_stats

def compute_bd_rate_over_anchor(stats, dataset_name):
    anchor_stats = get_anchor_stats(dataset_name)
    bdr = bd_rate(anchor_stats['bpp'], anchor_stats['psnr'], stats['bpp'], stats['psnr'])
    return bdr

def print_json_like(dict_of_list):
    for k, value in dict_of_list.items():
        if isinstance(value, list):
            vlist_str = '[' + ', '.join([f'{v:.12f}'[:6] for v in value]) + ']'
        else:
            vlist_str = value
        logging.info(f"'{k:<6s}': {vlist_str}")


def main():
    cfg = parse_args()
    TrainWrapper(cfg)


if __name__ == '__main__':
    main()
