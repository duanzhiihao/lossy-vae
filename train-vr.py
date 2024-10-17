from tqdm import tqdm
from collections import OrderedDict, defaultdict
import json
import logging
import argparse
import torch
from timm.utils import ModelEmaV3, unwrap_model, AverageMeter, random_seed
from accelerate import Accelerator

import mycv
import mycv.utils.training_v2 as mut
from mycv.eval import standard_evaluate

import lvae


def parse_args():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    boa = argparse.BooleanOptionalAction
    # wandb setting
    parser.add_argument("--wbproject",  type=str,   default="lattice-lic")
    parser.add_argument("--wbgroup",    type=str,   default="multi-rate")
    parser.add_argument("--wbmode",     type=str,   default="disabled")
    parser.add_argument("--run_name",   type=str,   default=None)
    # model setting
    parser.add_argument("--model",      type=str,   default="qarv_base")
    parser.add_argument("--model_args", type=str,   default="")
    # data setting
    parser.add_argument("--trainset",   type=str,   default="coco_train2017")
    parser.add_argument("--valset",     type=str,   default="kodak")
    # optimization setting
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--iterations", type=int,   default=1_000_000)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lr_sched",   type=str,   default="constant")
    parser.add_argument("--grad_clip",  type=float, default=2.0)
    # training acceleration and device setting
    parser.add_argument("--amp",        type=str,   default="no")
    parser.add_argument("--workers",    type=int,   default=8)
    cfg = parser.parse_args()

    # default settings
    cfg.wbnotes = f"{cfg.model_args}"
    cfg.wandb_log_interval = 20
    cfg.model_val_interval = 2_000
    return cfg


def get_model(cfg):
    kwargs = eval(f"dict({cfg.model_args})")
    model = lvae.get_model(cfg.model, **kwargs)

    logging.info(f"Model name = {cfg.model}, type = {type(model)}, args = {kwargs}")
    cfg.total_params = sum(p.numel() for p in model.parameters())
    cfg.learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters = {cfg.total_params/1e6:.2f}M, " + \
                 f"learnable parameters = {cfg.learnable_params/1e6:.2f}M \n")
    return model


def main():
    cfg = parse_args()

    mut.set_device()

    accelerator = Accelerator(mixed_precision=cfg.amp)
    device = accelerator.device
    is_main = accelerator.is_main_process
    random_seed(accelerator.process_index + 2)

    if is_main:
        log_dir, wbrun = mut.set_logging(cfg)

    model = get_model(cfg).to(device=device)

    # if True: # debug
    #     wpath = "runs/factorized/fcttny_a2_scpm_fourier_4/last_ema.pt"
    #     checkpoint = torch.load(wpath, map_location=device, weights_only=True)
    #     model.load_state_dict(checkpoint["model"])

    if is_main:
        model_ema = ModelEmaV3(model, use_warmup=True, warmup_power=3/4)

    optimizer = torch.optim.Adam(mut.learnable_params(model), lr=cfg.lr)

    model, optimizer = accelerator.prepare(model, optimizer)

    trainset = mycv.get_dataset(cfg.trainset)
    trainloader = mycv.datasets.make_train_loader(trainset, cfg.batch_size, cfg.workers)
    if is_main:
        val_img_dir = mycv.dataset_paths[cfg.valset]
        # valset = mycv.get_dataset(cfg.valset)
        # valloader = mycv.datasets.make_val_loader(valset, batch_size=1, workers=0)

    # ======================== training loops ========================
    pbar = range(0, cfg.iterations + 1)
    if is_main:
        pbar = tqdm(pbar, ascii=True)
    for step in pbar:
        log = OrderedDict()

        if step % 10 == 0: # learning rate schedule
            log["lr-factor"] = mut.adjust_lr(cfg, optimizer, step)

        # training step
        assert model.training
        batch = next(trainloader).to(device=device)
        metrics = model(batch)
        loss = metrics["loss"]

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        if is_main:
            model_ema.update(model, step=step)

            # logging
            metrics["grad"] = grad_norm.item()
            msg = ", ".join([f"{k}={float(v):.4f}" for k, v in metrics.items()])
            pbar.set_description(msg)

            if step % cfg.wandb_log_interval == 0: # log to wandb
                log = {f"general/{k}": v for k, v in log.items()}
                for i, pg in enumerate(optimizer.param_groups):
                    log[f"general/lr-pg{i}"] = pg["lr"]
                log["general/ema_decay"] = model_ema.get_decay(step)
                log.update({f"train/{k}": float(v) for k, v in metrics.items()})
                wbrun.log(log, step=step)

            if step % cfg.model_val_interval == 0:
                evaluate_and_log(model_ema, val_img_dir, log_dir, wbrun, step)
                if hasattr(model, "study"):
                    model.study(log_dir)
                mut.save_checkpoints(log_dir, step, model, model_ema, optimizer)
                model.train()

    logging.info("Training finished.")


def save_model(model, step, fpath):
    checkpoint = {"model": unwrap_model(model).state_dict(), "step": step}
    torch.save(checkpoint, f=fpath)


@torch.inference_mode()
def evaluate_and_log(model, val_img_dir, log_dir, wbrun, step):
    model = unwrap_model(model).eval()

    results = model.self_evaluate(val_img_dir, log_dir=log_dir, steps=6)
    print_json_like(results)

    results_to_log = OrderedDict({"general/iter": step})

    if step > 1000:
        anchor = _read_json("results/kodak/kodak-vtm18.0.json")
        bdr = mycv.utils.bd_rate(anchor['bpp'], anchor['psnr'], results['bpp'], results['psnr'])
        results_to_log['val-bd-rate/kodak-vtm18.0'] = bdr

    lambdas = results['lambda']
    for idx in range(len(lambdas)):
        lmb = round(lambdas[idx])
        results_to_log.update({
            f'lmb{lmb}/loss': results['loss'][idx],
            f'lmb{lmb}/bpp':  results['bpp'][idx],
            f'lmb{lmb}/psnr': results['psnr'][idx],
        })

    # wandb log
    wbrun.log(results_to_log, step=step)


def _read_json(fpath):
    with open(fpath, mode="r") as f:
        stats = json.load(fp=f)
    return stats.get("results", stats)

def print_json_like(dict_of_list):
    for k, value in dict_of_list.items():
        if isinstance(value, list):
            vlist_str = '[' + ', '.join([f'{v:.12f}'[:6] for v in value]) + ']'
        else:
            vlist_str = value
        logging.info(f"'{k:<6s}': {vlist_str}")


if __name__ == "__main__":
    main()
