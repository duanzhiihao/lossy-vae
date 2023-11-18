from tqdm import tqdm
import math
import argparse
import torch
from timm.utils import ModelEmaV2

import mycv
import lvae.utils.training as tr_utils


def parse_args():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    boa = argparse.BooleanOptionalAction
    # wandb setting
    parser.add_argument('--wbproject',  type=str,   default='qarv')
    parser.add_argument('--wbentity',   type=str,   default='prof-zhu-compression')
    parser.add_argument('--wbgroup',    type=str,   default='zd-qarv-tiny')
    parser.add_argument('--wbmode',     type=str,   default='disabled')
    parser.add_argument('--resume',     type=str,   default=None)
    # model setting
    parser.add_argument('--model',      type=str,   default='qv2_4z_z128')
    parser.add_argument('--model_args', type=str,   default='')
    # data setting
    parser.add_argument('--trainset',   type=str,   default='coco_train2017')
    parser.add_argument('--valset',     type=str,   default='kodak')
    # optimization setting
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--iterations', type=int,   default=500_000)
    parser.add_argument('--lr',         type=float, default=2e-4)
    parser.add_argument('--lr_sched',   type=str,   default='const-0.5-cos')
    parser.add_argument('--grad_clip',  type=float, default=2.0)
    parser.add_argument('--ema_decay',  type=float, default=0.9999)
    # training acceleration and device setting
    parser.add_argument('--compile',    action=boa, default=False)
    parser.add_argument('--workers',    type=int,   default=4)
    cfg = parser.parse_args()

    # default settings
    cfg.wandb_log_interval = 20
    cfg.model_val_interval = 2_000
    return cfg


def main():
    cfg = parse_args()

    log_dir, wbrun = tr_utils.set_logging(cfg)
    device = tr_utils.set_device()

    model = tr_utils.set_model(cfg)
    model = model.to(device)

    model_ema = ModelEmaV2(model, decay=cfg.ema_decay)
    model_cmp = torch.compile(model) if cfg.compile else model

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    trainset = mycv.get_dataset(cfg.trainset)
    trainloader = mycv.datasets.make_train_loader(trainset, cfg.batch_size, cfg.workers)
    valset = mycv.get_dataset(cfg.valset)
    valloader = mycv.datasets.make_val_loader(valset, batch_size=1, workers=0)

    # ======================== training loops ========================
    pbar = tqdm(range(0, cfg.iterations), ascii=True)
    for step in pbar:
        if (step > 0) and (step % cfg.model_val_interval == 0): # evaluation
            tr_utils.vr_evaluate_log(model_ema, valloader, wbrun, step)
            tr_utils.save_checkpoints(log_dir, step, model, model_ema, optimizer)
            model.train()

        if step % 10 == 0: # learning rate schedule
            tr_utils.adjust_lr(cfg, optimizer, step)

        # training step
        assert model_cmp.training
        batch = next(trainloader).to(device=device)
        metrics = model_cmp(batch)

        metrics['loss'].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model_cmp.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        model_ema.decay = cfg.ema_decay * (1 - math.exp(-step / 10_000)) # warmup from 0 to cfg.ema_decay
        model_ema.update(model)

        # logging
        metrics['grad'] = grad_norm.item()
        msg = ', '.join([f'{k}={float(v):.4f}' for k, v in metrics.items()])
        pbar.set_description(msg)
        if step % cfg.wandb_log_interval == 0: # log to wandb
            log_dict = {f'train/{k}': float(v) for k, v in metrics.items()}
            for i, pg in enumerate(optimizer.param_groups):
                log_dict[f'general/lr-pg{i}'] = pg['lr']
            wbrun.log(log_dict, step=step)

    # final evaluation
    tr_utils.vr_evaluate_log(model_ema, valloader, wbrun, step)
    tr_utils.save_checkpoints(log_dir, step, model, model_ema)


if __name__ == '__main__':
    main()
