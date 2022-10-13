import os
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict
import math
import torch
import torch.distributed
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision as tv
import wandb

import timm.utils
import lvae.utils as utils
import lvae.utils.ddp as ddputils

# import mycv.utils.loggers as mylog
# import mycv.utils as utils
# import mycv.utils.ddp as ddputils
# import mycv.utils.torch_utils as mytu
# import mycv.utils.lr_schedulers as lr_schedulers


def default_parser():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    # wandb setting
    parser.add_argument('--wbproject',  type=str,  default='default')
    parser.add_argument('--wbgroup',    type=str,  default=None)
    parser.add_argument('--wbtags',     type=str,  default=None, nargs='+')
    parser.add_argument('--wbnote',     type=str,  default=None)
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    parser.add_argument('--name',       type=str,  default=None)
    # model setting
    parser.add_argument('--model',      type=str,  default='model-name')
    parser.add_argument('--model_args', type=str,  default='')
    # resume setting
    parser.add_argument('--resume',     type=str,  default=None)
    parser.add_argument('--weights',    type=str,  default=None)
    parser.add_argument('--load_optim', action=argparse.BooleanOptionalAction, default=False)
    # data setting
    # parser.add_argument('--trainsets',  type=str,  default='imagenet64val')
    # parser.add_argument('--transform',  type=str,  default=None)
    # parser.add_argument('--valset',     type=str,  default='imagenet64val')
    # parser.add_argument('--val_bs',     type=int,  default=None)
    # optimization setting
    parser.add_argument('--batch_size', type=int,  default=16)
    parser.add_argument('--accum_num',  type=int,  default=1)
    parser.add_argument('--optimizer',  type=str,  default='adam')
    parser.add_argument('--lr',         type=float,default=1e-4)
    parser.add_argument('--lr_sched',   type=str,  default='constant')
    parser.add_argument('--lrf_min',    type=float,default=1e-2)
    parser.add_argument('--lr_warmup',  type=int,  default=0)
    parser.add_argument('--wdecay',     type=float,default=0.0)
    parser.add_argument('--grad_clip',  type=float,default=200.0)
    # parser.add_argument('--epochs',     type=int,  default=1000)
    # parser.add_argument('--iterations', type=int,  default=1_000_000)
    # automatic mixed precision (AMP)
    parser.add_argument('--amp',        action=argparse.BooleanOptionalAction, default=False)
    # exponential moving averaging (EMA)
    parser.add_argument('--ema',        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ema_decay',  type=float,default=0.9999)
    parser.add_argument('--ema_warmup', type=int,  default=None)
    # logging setting
    parser.add_argument('--eval_first', action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument('--eval_per',   type=int,  default=1)
    # device setting
    parser.add_argument('--fixseed',    action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--workers',    type=int,  default=0)
    return parser


class BaseTrainWrapper():
    # override these values in the child class
    model_registry_group: str
    wandb_log_interval = 100
    model_log_interval = wandb_log_interval
    grad_norm_interval = 100

    def __init__(self) -> None:
        # self._epoch_len: int
        self._cur_epoch = 0
        self._cur_iter  = 0
        self._best_loss = math.inf
        # miscellaneous
        self._moving_grad_norm_buffer = utils.MaxLengthList(max_len=self.grad_norm_interval)
        self.wandb_log_keys = set()

        # pytorch DDP setting
        self.local_rank  = int(os.environ.get('LOCAL_RANK', -1))
        self.world_size  = int(os.environ.get('WORLD_SIZE', 1))
        self.distributed = (self.world_size > 1)
        self.is_main     = self.local_rank in (-1, 0)

    def main(self, cfg):
        self.cfg = cfg

        # preparation
        self.set_logging()
        self.set_device()
        self.prepare_configs()
        if self.distributed:
            with ddputils.run_zero_first(): # training set
                self.set_dataset_()
            torch.distributed.barrier()
        else:
            self.set_dataset_()
        self.set_model_()
        self.set_optimizer_()
        self.set_pretrain()

        # logging
        self.ema = None
        if self.is_main:
            self.set_wandb_()
            self.set_ema_()
            header = ['Epoch', 'Iter', 'GPU_mem', 'lr', 'grad']
            self.stats_table = utils.SimpleTable(header)

        if self.distributed: # DDP mode
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # the main training loops
        self.training_loops()

    def set_logging(self):
        cfg = self.cfg

        # set logging
        if self.is_main: # main process
            print()
        else: # subprocess spawned by pytorch DDP
            mylog.reset_ddp_setting()

        # set log directory
        log_parent = Path(f'runs/{cfg.wbproject}').resolve()
        if cfg.resume is not None: # resume
            log_dir = log_parent / str(cfg.resume)
            assert log_dir.is_dir(), f'Try to resume from {log_dir} but it does not exist'
        elif self.is_main: # new experiment, main process
            run_name = utils.increment_dir(log_parent, name=cfg.model) if (cfg.name is None) else cfg.name
            log_dir = log_parent / str(run_name)
            os.makedirs(log_dir, exist_ok=False)
            utils.json_dump(cfg.__dict__, fpath=log_dir / 'config.json')
        else: # new experiment, ddp processes
            log_dir = None

        _dir_str = utils.ANSI.colorstr(str(log_dir), c='br_b', ul=True)
        _prefix = 'Resuming' if cfg.resume else 'Logging'
        logging.info(f'{_prefix} run at {_dir_str} \n')

        self.cfg.log_dir = str(log_dir)
        self._log_dir = log_dir

    def set_device(self):
        local_rank = self.local_rank
        world_size = self.world_size

        torch.cuda.set_device(max(local_rank, 0))
        torch.cuda.empty_cache()
        _count = torch.cuda.device_count()

        if world_size == 1: # standard single GPU mode
            assert (local_rank == -1) and self.is_main
            logging.info(f'Visible devices={_count}, using idx 0: {torch.cuda.get_device_properties(0)} \n')
            local_rank = 0 # just for selecting device 0
            mylog.add_file_handler(fpath=self._log_dir / f'logs.txt')
        else: # DDP mode
            assert torch.distributed.is_nccl_available()
            torch.distributed.init_process_group(backend="nccl")
            assert local_rank == torch.distributed.get_rank()
            assert world_size == torch.distributed.get_world_size()

            # communicate log_dir
            log_dir = ddputils.broadcast_object(str(self._log_dir), src=0, local_rank=local_rank)
            self._log_dir = Path(log_dir)
            mylog.add_file_handler(fpath=self._log_dir / f'logs_rank{local_rank}.txt')

            with ddputils.run_sequentially():
                msg = f'local_rank={local_rank}, world_size={world_size}, total visible={_count}'
                mylog.log(mylog.FORCE, msg)
                mylog.log(mylog.FORCE, f'{torch.cuda.get_device_properties(local_rank)} \n')
            torch.distributed.barrier()

        self.device = torch.device('cuda', local_rank)

    def prepare_configs(self):
        cfg = self.cfg

        if cfg.fixseed: # fix random seeds for reproducibility
            timm.utils.random_seed(2 + self.local_rank)
        torch.backends.cudnn.benchmark = True

        logging.info(f'Batch size on each dataloader (ie, GPU) = {cfg.batch_size}')
        logging.info(f'Gradient accmulation: {cfg.accum_num} backwards() -> one step()')
        bs_effective = cfg.batch_size * self.world_size * cfg.accum_num
        msg = f'Effective batch size = {bs_effective}, learning rate = {cfg.lr}, ' + \
              f'weight decay = {cfg.wdecay}'
        logging.info(msg)
        lr_per_1024img = cfg.lr / bs_effective * 1024
        logging.info(f'Learning rate per 1024 images = {lr_per_1024img}')
        wd_per_1024img = cfg.wdecay / bs_effective * 1024
        logging.info(f'Weight decay per 1024 images = {wd_per_1024img} \n')
        logging.info(f'Training config: \n{cfg} \n')

        cfg.bs_effective = bs_effective
        cfg.world_size = self.world_size

    def set_dataset_(self):
        self._epoch_len: int

    def set_model_(self):
        cfg = self.cfg

        assert hasattr(self, 'model_registry_group')
        from mycv.models.registry import get_registerd_model
        _model_func = get_registerd_model(self.model_registry_group, cfg.model)
        kwargs = eval(f'dict({cfg.model_args})')
        model = _model_func(**kwargs)
        assert isinstance(model, torch.nn.Module)

        cfg.num_param = sum([p.numel() for p in model.parameters() if p.requires_grad])
        logging.info(f'Using model name={cfg.model}, {type(model)}, args = {kwargs}')
        logging.info(f'Number of learnable parameters = {cfg.num_param/1e6} M \n')
        if self.is_main:
            utils.print_to_file(str(model), fpath=self._log_dir / 'model.txt', mode='w')

        self.model = model.to(self.device)

    def set_optimizer_(self):
        cfg, model = self.cfg, self.model

        # different optimization setting for different layers
        pgb, pgw, pgo = [], [], []
        pg_info = defaultdict(list)
        for k, v in model.named_parameters():
            assert isinstance(k, str) and isinstance(v, torch.Tensor)
            if not v.requires_grad:
                continue
            if ('.bn' in k) or ('.bias' in k): # batchnorm or bias
                pgb.append(v)
                pg_info['bn/bias'].append(f'{k:<80s} {v.shape}')
            elif '.weight' in k: # conv or linear weights
                pgw.append(v)
                pg_info['weights'].append(f'{k:<80s} {v.shape}')
            else: # other parameters
                pgo.append(v)
                pg_info['other'].append(f'{k:<80s} {v.shape}')
        parameters = [
            {'params': pgw, 'lr': cfg.lr, 'weight_decay': cfg.wdecay},
            {'params': pgb, 'lr': cfg.lr, 'weight_decay': 0.0},
            {'params': pgo, 'lr': cfg.lr, 'weight_decay': 0.0}
        ]
        # logging
        for pg in parameters:
            num_, lr_, wd_ = len(pg['params']), pg['lr'], pg['weight_decay']
            msg = f'num={num_:<4}, lr={lr_}, weight_decay={wd_}'
            pg_info['groups'].append(msg)
            logging.info(msg)
        msg = ', '.join([f'[{k}: {len(pg)}]' for k, pg in pg_info.items()])
        logging.info(f'optimizer parameter groups: {msg} \n')
        if self.is_main:
            utils.json_dump(pg_info, fpath=self._log_dir / 'optimizer.json')

        # optimizer
        if cfg.optimizer == 'sgd':
            cfg.sgd_momentum = 0.9
            optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.sgd_momentum)
        elif cfg.optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=cfg.lr)
        elif cfg.optimizer == 'adamax':
            optimizer = torch.optim.Adamax(parameters, lr=cfg.lr)
        else:
            raise ValueError(f'Unknown optimizer: {cfg.optimizer}')

        self.optimizer = optimizer
        self.scaler = amp.GradScaler(enabled=cfg.amp) # Automatic mixed precision

    def adjust_lr_(self, t, T):
        cfg = self.cfg

        # learning rate warm-up to prevent gradient exploding in early stages
        T_warm = cfg.lr_warmup
        if t < T_warm:
            lrf = min(t + 1, T_warm) / T_warm
        elif cfg.lr_sched == 'constant':
            lrf = 1.0
        elif cfg.lr_sched == 'cosine':
            lrf = lr_schedulers.get_cosine_lrf(t-T_warm, cfg.lrf_min, T-T_warm-1)
        elif cfg.lr_sched == 'const-0.75-cos':
            boundary = round(T * 0.75)
            if t <= boundary:
                lrf = 1.0
            else:
                lrf = lr_schedulers.get_cosine_lrf(t-boundary, cfg.lrf_min, T-boundary-1)
        else:
            raise NotImplementedError(f'cfg.lr_sched = {cfg.lr_sched} not implemented')

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cfg.lr * lrf

    def set_pretrain(self):
        cfg = self.cfg

        if cfg.resume is not None: # resume
            assert not cfg.weights, f'--resume={cfg.resume} not compatible with --weights={cfg.weights}'
            ckpt_path = self._log_dir / 'last.pt'
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cfg.lr
            results = checkpoint.get('results', dict())
            self._cur_iter  = checkpoint['iter']
            self._cur_epoch = checkpoint['epoch']
            self._best_loss = results.get('loss', self._best_loss)
            logging.info(f'Loaded checkpoint from {ckpt_path}. results={results}, '
                         f'Epoch={self._cur_epoch}, iterations={self._cur_iter} \n')
        elif cfg.weights is not None: # (partially or fully) initialize from pretrained weights
            checkpoint = torch.load(cfg.weights, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'], strict=False)
            if cfg.load_optim:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scaler.load_state_dict(checkpoint['scaler'])
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cfg.lr
            logging.info(f'Loaded checkpoint from {cfg.weights}. optimizer={cfg.load_optim}. \n')
        else:
            logging.info('No pre-trained weights provided. Will train from scratch. \n')

    def set_wandb_(self):
        cfg = self.cfg

        # check if there is a previous run to resume
        wbid_path = self._log_dir / 'wandb_id.txt'
        rid = utils.read_file(wbid_path).strip().split('\n')[-1] if wbid_path.is_file() else None
        # initialize wandb
        run_name = self._log_dir.stem
        if cfg.wbnote is not None:
            run_name = f'{run_name}: {cfg.wbnote}'
        wbrun = wandb.init(
            project=cfg.wbproject, group=cfg.wbgroup, name=run_name, tags=cfg.wbtags,
            config=cfg, dir='runs/', id=rid, resume='allow', save_code=True, mode=cfg.wbmode
        )
        cfg = wbrun.config
        cfg.wandb_id = wbrun.id
        utils.print_to_file(wbrun.id, fpath=wbid_path, mode='a')

        self.wbrun = wbrun
        self.cfg = cfg

    def set_ema_(self):
        # Exponential moving averaging (EMA)
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        cfg = self.cfg

        if cfg.ema:
            from timm.utils.model_ema import ModelEmaV2 # lazy import
            ema = ModelEmaV2(self.model, decay=cfg.ema_decay)

            msg = f'Using EMA with decay={cfg.ema_decay}.'
            if cfg.resume:
                ckpt_path = self._log_dir / 'last_ema.pt'
                assert ckpt_path.is_file(), f'Cannot find EMA checkpoint: {ckpt_path}'
                ema.module.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
                msg = msg + f' Loaded EMA from {ckpt_path}.'
            logging.info(msg + '\n')
        else:
            ema = None

        self.ema = ema

    def training_loops(self):
        raise NotImplementedError()

    def gradient_clip_(self, parameters):
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.cfg.grad_clip)
        # grad_norm = mytu.get_grad_norm(parameters)
        self._moving_grad_norm_buffer.add(float(grad_norm))
        moving_median = self._moving_grad_norm_buffer.median()
        # _clip = self.cfg.grad_clip
        if grad_norm > (moving_median * 10): # super large gradient
            # _clip = min(moving_median, _clip) / 10
            # mylog.warning(f'Large gradient norm = {grad_norm:3f}. Clipping to {_clip:.3f} ...')
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1
            _lr = param_group['lr']
            logging.warning(f'Large gradient norm = {grad_norm:3f}. Set lr={_lr} .')
            if self.is_main and (grad_norm > self.cfg.grad_clip * 20):
                checkpoint = {'model': timm.utils.unwrap_model(self.model).state_dict()}
                wpath = self._log_dir / 'bad.pt'
                torch.save(checkpoint, wpath)
                logging.warning(f'Saved bad model to {wpath}. Please debug it.')
            bad = True
        else:
            bad = False
        # mytu.clip_grad_norm_(parameters, max_norm=_clip, computed_norm=grad_norm)
        return grad_norm, bad

    def init_logging_(self, print_header=True):
        assert self.is_main
        print()
        # initialize stats table and progress bar
        for k in self.stats_table.keys():
            self.stats_table[k] = 0.0
        self._pbar_header = self.stats_table.get_header(border=True)
        if print_header:
            print(self._pbar_header)
        time.sleep(0.1)

    @torch.no_grad()
    def minibatch_log(self, pbar, epoch, bi, grad_norm, stats):
        assert self.is_main, f'is_main={self.is_main}, local_rank={self.local_rank}'
        cfg = self.cfg

        n = len(str(cfg.epochs-1))
        self.stats_table['Epoch'] = f'{epoch:>{n}}/{cfg.epochs-1}'
        global_step = self._epoch_len * epoch + bi
        n = len(str(cfg.iterations))
        self.stats_table['Iter']  = f'{global_step:>{n}}/{cfg.iterations}'

        mem = torch.cuda.max_memory_allocated(self.device) / 1e9
        torch.cuda.reset_peak_memory_stats()
        self.stats_table['GPU_mem'] = f'{mem:.3g}G'

        cur_lr = self.optimizer.param_groups[0]['lr']
        self.stats_table['lr'] = cur_lr

        self._moving_max_grad_norm = max(self._moving_max_grad_norm, grad_norm)
        self.stats_table['grad'] = grad_norm

        for k,v in stats.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = float(v.detach().cpu().item())
            assert isinstance(v, (float, int))
            prev = self.stats_table.get(k, 0.0)
            self.stats_table[k] = (prev * bi + v) / (bi + 1)
            self.wandb_log_keys.add(k)

        pbar_header, pbar_body = self.stats_table.update(border=True)
        if pbar_header != self._pbar_header: # update the progress bar header
            print(pbar_header)
            self._pbar_header = pbar_header
        pbar.set_description(pbar_body)

    @torch.no_grad()
    def periodic_log(self, batch):
        # model logging
        if self._cur_iter % self.model_log_interval == 0:
            self.model.eval()
            _model = timm.utils.unwrap_model(self.model)
            if hasattr(_model, 'study'):
                _model.study(save_dir=self._log_dir, wandb_run=self.wbrun)
            self.model.train()

        # Weights & Biases logging
        if self._cur_iter % self.wandb_log_interval == 0:
            imgs = batch if torch.is_tensor(batch) else batch[0]
            assert torch.is_tensor(imgs)
            N = min(16, imgs.shape[0])
            tv.utils.save_image(imgs[:N], fp=self._log_dir / 'inputs.png', nrow=math.ceil(N**0.5))

            _log_dic = {
                'general/lr': self.optimizer.param_groups[0]['lr'],
                'general/grad_norm': self._moving_max_grad_norm,
                # 'ema/n_updates': (self.ema.updates if self.ema else 0),
                # 'ema/decay': (self.ema.get_decay() if self.ema else 0)
                'ema/decay': (self.ema.decay if self.ema else 0)
            }
            _log_dic.update(
                {'train/'+k: self.stats_table[k] for k in self.wandb_log_keys}
            )
            self.wbrun.log(_log_dic, step=self._cur_iter)
            # reset running max grad norm
            self._moving_max_grad_norm = 0.0

    def eval_model(self, model) -> dict:
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self):
        assert self.is_main
        # Evaluation
        _log_dic = {
            'general/epoch': self._cur_epoch,
            'general/iter':  self._cur_iter
        }
        model_ = timm.utils.unwrap_model(self.model).eval()
        results = self.eval_model(model_)
        mylog.log_dict_as_table(results)
        _log_dic.update({'val-metrics/plain_'+k: v for k,v in results.items()})
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
        torch.save(checkpoint, self._log_dir / 'last.pt')
        self._save_if_best(checkpoint)

        if self.cfg.ema:
            no_ema_loss = results['loss']
            results = self.eval_model(self.ema.module)
            mylog.log_dict_as_table(results)
            _log_dic.update({f'val-metrics/ema_'+k: v for k,v in results.items()})
            # save last checkpoint of EMA
            checkpoint = {
                'model': self.ema.module.state_dict(),
                'epoch': self._cur_epoch,
                'iter':  self._cur_iter,
                'results' : results,
            }
            torch.save(checkpoint, self._log_dir / 'last_ema.pt')
            self._save_if_best(checkpoint)

        # wandb log
        self.wbrun.log(_log_dic, step=self._cur_iter)
        # Log evaluation results to file
        msg = self.stats_table.get_body() + '||' + '%10.4g' * 1 % (results['loss'])
        with open(self._log_dir / 'results.txt', 'a') as f:
            f.write(msg + '\n')

        self._results = results
        print()

    def _save_if_best(self, checkpoint):
        assert self.is_main
        # save checkpoint if it is the best so far
        cur_loss = checkpoint['results']['loss']
        if cur_loss < self._best_loss:
            self._best_loss = cur_loss
            svpath = self._log_dir / 'best.pt'
            torch.save(checkpoint, svpath)
            logging.info(f'Get best loss = {cur_loss}. Saved to {svpath}.')

    def clean_and_exit(self):
        logging.error(f'Terminating local rank {self.local_rank}...')
        if self.is_main: # save failed checkpoint for debugging
            checkpoint = {'model': timm.utils.unwrap_model(self.model).state_dict()}
            torch.save(checkpoint, self._log_dir / 'failed.pt')
        if self.distributed:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            time.sleep(0.2 * self.local_rank)
        exit(utils.ANSI.sccstr(f'Local rank {self.local_rank} safely terminated.'))


class IterTrainWrapper(BaseTrainWrapper):
    def prepare_configs(self):
        super().prepare_configs()
        self._log_ema_weight = 5.0 / (self.cfg.log_itv + 8.0)
        _m = 1-self._log_ema_weight
        msg = f'train metrics avg weight={self._log_ema_weight:.4f}, momentum={_m:.4f} \n'
        logging.info(msg)

    @torch.no_grad()
    def minibatch_log(self, pbar, stats):
        assert self.is_main, f'is_main={self.is_main}, local_rank={self.local_rank}'
        cfg = self.cfg

        epoch = float(self._cur_iter / self._epoch_len)
        self.stats_table['Epoch'] = f'{epoch:.1f}/{cfg.epochs:.1f}'
        n = len(str(cfg.iterations))
        self.stats_table['Iter'] = f'{self._cur_iter:>{n}}/{cfg.iterations-1}'

        mem = torch.cuda.max_memory_allocated(self.device) / 1e9
        torch.cuda.reset_peak_memory_stats()
        self.stats_table['GPU_mem'] = f'{mem:.3g}G'

        cur_lr = self.optimizer.param_groups[0]['lr']
        self.stats_table['lr'] = cur_lr

        self.stats_table['grad'] = self._moving_grad_norm_buffer.current()

        for k,v in stats.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = float(v.detach().cpu().item())
            assert isinstance(v, (float, int))
            prev = self.stats_table.get(k, 0.0)
            if prev == 0.0:
                new = v
            else: # exponential moving average
                assert self.wandb_log_interval >= 2
                new = (1 - self._log_ema_weight) * prev + self._log_ema_weight * v
            self.stats_table[k] = new
            self.wandb_log_keys.add(k)

        pbar_header, pbar_body = self.stats_table.update(border=True)
        if pbar_header != self._pbar_header: # update the progress bar header
            print(pbar_header)
            self._pbar_header = pbar_header
        pbar.set_description(pbar_body)
