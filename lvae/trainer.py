from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import os
import time
import logging
import math
import torch
import torch.distributed
import torch.cuda.amp as amp
import torchvision as tv
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.utils import ModelEmaV2, unwrap_model, random_seed

import lvae.utils as utils
from lvae.models.registry import get_model


class BaseTrainingWrapper():
    # overwrite these values in the child class if necessary
    grad_norm_interval = 100

    def __init__(self, cfg):
        self.cfg = cfg

        # initialize the training status
        self._cur_epoch = 0
        self._cur_iter  = 0
        self._best_loss = math.inf
        # miscellaneous
        self._moving_grad_norm_buffer = utils.MaxLengthList(max_len=self.grad_norm_interval)
        self.wandb_log_keys = set()

        # progress bar logging
        header = ['Epoch', 'Iter', 'GPU_mem', 'lr', 'grad']
        self.stats_table = utils.SimpleTable(header)

        # pytorch DDP setting
        self.local_rank  = int(os.environ.get('LOCAL_RANK', -1))
        self.world_size  = int(os.environ.get('WORLD_SIZE', 1))
        self.distributed = (self.world_size > 1)
        self.is_main     = self.local_rank in (-1, 0)

    def main(self):
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

    def set_logging(self):
        cfg = self.cfg

        # set logging
        if self.is_main: # main process
            print()
            handler = utils.my_stream_handler()
            logging.basicConfig(handlers=[handler], level=logging.INFO)
        else: # subprocess spawned by pytorch DDP
            fmt = f'[%(asctime)s RANK={self.local_rank}] [%(levelname)s] %(message)s'
            logging.basicConfig(format=fmt, level=logging.WARNING)

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
        local_rank = max(self.local_rank, 0)
        world_size = self.world_size

        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        _count = torch.cuda.device_count()
        _info = torch.cuda.get_device_properties(local_rank)

        if world_size == 1: # standard single GPU mode
            assert (local_rank == 0) and self.is_main
            logging.info(f'Total {_count} visible devices, using idx 0: {_info} \n')
        else: # DDP mode
            assert torch.distributed.is_nccl_available()
            torch.distributed.init_process_group(backend="nccl")
            assert local_rank == torch.distributed.get_rank()
            assert world_size == torch.distributed.get_world_size()

            time.sleep(0.1 * local_rank)
            print(f'local_rank={local_rank}, world_size={world_size} \n{_info} \n')
            torch.distributed.barrier()

        self.device = torch.device('cuda', local_rank)

    def prepare_configs(self):
        cfg = self.cfg

        if cfg.fixseed: # fix random seeds for reproducibility
            random_seed(2 + self.local_rank)
        torch.backends.cudnn.benchmark = True

        logging.info(f'Batch size on each GPU = {cfg.batch_size}')
        logging.info(f'Gradient accmulation: {cfg.accum_num} backwards() -> one optimizer.step()')
        bs_effective = cfg.batch_size * self.world_size * cfg.accum_num
        msg = f'Effective batch size = {bs_effective}, learning rate = {cfg.lr}, ' + \
              f'weight decay = {cfg.wdecay} \n'
        logging.info(msg)
        logging.info(f'Training config: \n{cfg} \n')

        cfg.bs_effective = bs_effective
        cfg.world_size = self.world_size

        self._log_ema_weight = 5.0 / (cfg.wandb_log_interval + 8.0)
        _m = 1 - self._log_ema_weight
        msg = f'train metrics avg weight={self._log_ema_weight:.4f}, momentum={_m:.4f} \n'
        logging.info(msg)

    def set_dataset(self):
        self._epoch_len: int

    def set_model(self):
        cfg = self.cfg

        kwargs = eval(f'dict({cfg.model_args})')
        model = get_model(cfg.model, **kwargs)
        assert isinstance(model, torch.nn.Module)

        cfg.num_param = sum([p.numel() for p in model.parameters() if p.requires_grad])
        logging.info('==== Model ====')
        logging.info(f'Model name = {cfg.model}, type = {type(model)}, args = {kwargs}')
        logging.info(f'Number of learnable parameters = {cfg.num_param/1e6} M \n')
        if self.is_main:
            utils.print_to_file(str(model), fpath=self._log_dir / 'model.txt', mode='w')

        self.model = model.to(self.device)

    def set_optimizer(self):
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
        logging.info('==== Optimizer ====')
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
            optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=0.9, weight_decay=cfg.wdecay)
        elif cfg.optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.wdecay)
        elif cfg.optimizer == 'adamax':
            optimizer = torch.optim.Adamax(parameters, lr=cfg.lr, weight_decay=cfg.wdecay)
        else:
            raise ValueError(f'Unknown optimizer: {cfg.optimizer}')

        self.optimizer = optimizer
        self.scaler = amp.GradScaler(enabled=cfg.amp) # Automatic mixed precision

    @staticmethod
    def get_cosine_factor(t, T, final=0.01):
        """ As `t` goes from `0` to `T`, return value goes from `1` to `final`
        """
        return final + 0.5 * (1 - final) * (1 + math.cos(t * math.pi / T))

    def adjust_lr(self, t, T):
        cfg = self.cfg

        # learning rate warm-up to prevent gradient exploding in early stages
        T_warm = cfg.lr_warmup
        if t < T_warm:
            lrf = (t + 1) / T_warm
        elif cfg.lr_sched == 'constant':
            lrf = 1.0
        elif cfg.lr_sched == 'cosine':
            lrf = self.get_cosine_factor(t-T_warm, T-T_warm-1, final=cfg.lrf_min)
        elif cfg.lr_sched == 'const-0.5-cos': # constant LR (50% training) + cosine LR (50% training)
            boundary = round(T * 0.5)
            if t <= boundary:
                lrf = 1.0
            else:
                lrf = self.get_cosine_factor(t-boundary, T-boundary-1, final=cfg.lrf_min)
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

    def set_wandb(self):
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

    def set_ema(self):
        # Exponential moving averaging (EMA)
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        cfg = self.cfg

        if cfg.ema:
            ema = ModelEmaV2(self.model, decay=cfg.ema_decay)

            msg = f'Training uses EMA with decay = {cfg.ema_decay}.'
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
                if cfg.model_val_interval <= 0: # no evaluation
                    pass
                elif (step == 0) and (not cfg.eval_first): # first iteration
                    pass
                elif step % cfg.model_val_interval == 0: # evaluaion
                    self.evaluate()
                    model.train()
                    print(self._pbar_header)

            # learning rate schedule
            if step % 10 == 0:
                self.adjust_lr(step, cfg.iterations)

            # DDP sampler: make sure each epoch has different random seed
            if self.distributed:
                self.trainsampler.set_epoch(step)

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

    def gradient_clip(self, parameters):
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.cfg.grad_clip)
        self._moving_grad_norm_buffer.add(float(grad_norm))
        moving_median = self._moving_grad_norm_buffer.median()
        if grad_norm > (moving_median * 10): # super large gradient
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1
            _lr = param_group['lr']
            logging.warning(f'Large gradient norm = {grad_norm:3f}. Set lr={_lr} .')
            bad = True
        else:
            bad = False
        return grad_norm, bad

    def init_progress_table(self):
        assert self.is_main
        print()
        # initialize stats table and progress bar
        for k in self.stats_table.keys():
            self.stats_table[k] = 0.0
        self._pbar_header = self.stats_table.get_header(border=True)
        time.sleep(0.1)

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
                assert cfg.wandb_log_interval >= 2
                new = (1 - self._log_ema_weight) * prev + self._log_ema_weight * v
            self.stats_table[k] = new
            self.wandb_log_keys.add(k)

        pbar_header, pbar_body = self.stats_table.update(border=True)
        if len(pbar_header) != len(self._pbar_header): # update the progress bar header
            print(pbar_header)
            self._pbar_header = pbar_header
        pbar.set_description(pbar_body)

    @torch.no_grad()
    def periodic_log(self, batch):
        # model logging
        if self._cur_iter % self.cfg.model_log_interval == 0:
            self.model.eval()
            _model = unwrap_model(self.model)
            if hasattr(_model, 'study'):
                _model.study(save_dir=self._log_dir, wandb_run=self.wbrun)
            self.model.train()

        # Weights & Biases logging
        if self._cur_iter % self.cfg.wandb_log_interval == 0:
            imgs = batch if torch.is_tensor(batch) else batch[0]
            assert torch.is_tensor(imgs)
            N = min(16, imgs.shape[0])
            tv.utils.save_image(imgs[:N], fp=self._log_dir / 'inputs.png', nrow=math.ceil(N**0.5))

            _log_dic = {
                'general/lr': self.optimizer.param_groups[0]['lr'],
                'general/grad_norm': self._moving_grad_norm_buffer.max(),
                'ema/decay': (self.ema.decay if self.ema else 0)
            }
            _log_dic.update(
                {'train/'+k: self.stats_table[k] for k in self.wandb_log_keys}
            )
            self.wbrun.log(_log_dic, step=self._cur_iter)

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
        model_ = unwrap_model(self.model).eval()
        results = self.eval_model(model_)
        logging.info(f'Validation results (no EMA): {results}')
        utils.print_dict_as_table(results)
        _log_dic.update({'val-metrics/plain_'+k: v for k,v in results.items()})
        # save last checkpoint
        checkpoint = {
            'model'     : model_.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'scaler'    : self.scaler.state_dict(),
            'epoch': self._cur_epoch,
            'iter':  self._cur_iter,
            'results'   : results,
        }
        torch.save(checkpoint, self._log_dir / 'last.pt')
        self._save_if_best(checkpoint)

        if self.cfg.ema:
            results = self.eval_model(self.ema.module.eval())
            logging.info(f'Validation results (EMA): {results}')
            utils.print_dict_as_table(results)
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
            checkpoint = {'model': unwrap_model(self.model).state_dict()}
            torch.save(checkpoint, self._log_dir / 'failed.pt')
        if self.distributed:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            time.sleep(0.2 * self.local_rank)
        exit(utils.ANSI.sccstr(f'Local rank {self.local_rank} safely terminated.'))
