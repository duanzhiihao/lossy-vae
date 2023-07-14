import json
import logging
import argparse
import torch
from timm.utils import unwrap_model

from lvae.utils.coding import bd_rate
from lvae.paths import known_datasets
from lvae.trainer import BaseTrainingWrapper
from lvae.datasets import get_image_dateset, make_trainloader


def parse_args():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    # wandb setting
    parser.add_argument('--wbproject',  type=str,  default='qarv')
    parser.add_argument('--wbentity',   type=str,  default='prof-zhu-compression')
    parser.add_argument('--wbgroup',    type=str,  default='var-rate-exp')
    parser.add_argument('--wbtags',     type=str,  default=None, nargs='+')
    parser.add_argument('--wbnote',     type=str,  default=None)
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    parser.add_argument('--name',       type=str,  default=None)
    # model setting
    parser.add_argument('--model',      type=str,  default='qarv_base')
    parser.add_argument('--model_args', type=str,  default='')
    # resume setting
    parser.add_argument('--resume',     type=str,  default=None)
    parser.add_argument('--weights',    type=str,  default=None)
    parser.add_argument('--load_optim', action=argparse.BooleanOptionalAction, default=False)
    # data setting
    parser.add_argument('--trainset',   type=str,  default='coco-train2017')
    parser.add_argument('--transform',  type=str,  default='crop=256,hflip=True')
    parser.add_argument('--valset',     type=str,  default='kodak')
    parser.add_argument('--val_steps',  type=int,  default=8)
    # optimization setting
    parser.add_argument('--batch_size', type=int,  default=32)
    parser.add_argument('--accum_num',  type=int,  default=1)
    parser.add_argument('--optimizer',  type=str,  default='adam')
    parser.add_argument('--lr',         type=float,default=2e-4)
    parser.add_argument('--lr_sched',   type=str,  default='const-0.5-cos')
    parser.add_argument('--lrf_min',    type=float,default=0.01)
    parser.add_argument('--lr_warmup',  type=int,  default=0)
    parser.add_argument('--grad_clip',  type=float,default=2.0)
    # training iterations setting
    parser.add_argument('--iterations', type=int,  default=2_000_000)
    parser.add_argument('--eval_first', action=argparse.BooleanOptionalAction, default=False)
    # exponential moving averaging (EMA)
    parser.add_argument('--ema',        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ema_decay',  type=float,default=0.9999)
    parser.add_argument('--ema_warmup', type=int,  default=10_000)
    # device setting
    parser.add_argument('--fixseed',    action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--workers',    type=int,  default=8)
    cfg = parser.parse_args()

    # default settings
    cfg.wdecay = 0.0
    cfg.amp = False
    cfg.wandb_log_interval = 100
    cfg.model_log_interval = 2000
    cfg.model_val_interval = 2000
    return cfg


class TrainWrapper(BaseTrainingWrapper):
    def set_dataset(self):
        cfg = self.cfg

        logging.info('==== Datasets and Dataloaders ====')
        trainset = get_image_dateset(cfg.trainset, transform_cfg=cfg.transform)
        self.make_training_loader(trainset)

        logging.info(f'Training root: {trainset.root}')
        logging.info(f'Number of training images = {len(trainset)}')
        logging.info(f'Training transform: \n{str(trainset.transform)}')
        logging.info(f'Validation root: {known_datasets[cfg.valset]} \n')

    @torch.no_grad()
    def evaluate(self):
        assert self.is_main
        log_dir = self._log_dir
        cfg = self.cfg
        val_img_dir = known_datasets[cfg.valset]

        # Evaluation
        _log_dic = {
            'general/epoch': self._cur_epoch,
            'general/iter':  self._cur_iter
        }
        model_ = unwrap_model(self.model).eval()
        results = model_.self_evaluate(val_img_dir, log_dir=log_dir, steps=cfg.val_steps)
        results_to_log = process_log_results(results, cfg.valset)

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

        if cfg.ema:
            results = self.ema.module.self_evaluate(val_img_dir, log_dir=log_dir, steps=cfg.val_steps)
            results_to_log = process_log_results(results, cfg.valset)
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


def process_log_results(results, dataset_name='kodak'):
    bdr = compute_bd_rate_over_anchor(results, dataset_name)
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
        'kodak': 'results/kodak/kodak-vtm18.0.json',
        'tecnick-rgb-1200': 'results/tecnick-rgb-1200/tecnick-rgb-1200-vtm18.0.json',
        'clic2022-test': 'results/clic2022-test/clic2022-test-vtm18.0.json'
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
    trainer = TrainWrapper(cfg)
    trainer.main()


if __name__ == '__main__':
    main()
