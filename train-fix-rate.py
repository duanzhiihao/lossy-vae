import logging
import argparse

from lvae.trainer import BaseTrainingWrapper
from lvae.datasets.image import get_image_dateset
from lvae.evaluation import image_self_evaluate


def parse_args():
    # ====== set the run settings ======
    parser = argparse.ArgumentParser()
    # wandb setting
    parser.add_argument('--wbproject',  type=str,  default='default')
    parser.add_argument('--wbentity',   type=str,  default=None)
    parser.add_argument('--wbgroup',    type=str,  default='fix-rate-exp')
    parser.add_argument('--wbtags',     type=str,  default=None, nargs='+')
    parser.add_argument('--wbnote',     type=str,  default=None)
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    parser.add_argument('--name',       type=str,  default=None)
    # model setting
    parser.add_argument('--model',      type=str,  default='qres34m')
    parser.add_argument('--model_args', type=str,  default='lmb=2048')
    # resume setting
    parser.add_argument('--resume',     type=str,  default=None)
    parser.add_argument('--weights',    type=str,  default=None)
    parser.add_argument('--load_optim', action=argparse.BooleanOptionalAction, default=False)
    # data setting
    parser.add_argument('--trainset',   type=str,  default='coco-train2017')
    parser.add_argument('--transform',  type=str,  default='crop=256,hflip=True')
    parser.add_argument('--valset',     type=str,  default='kodak')
    # optimization setting
    parser.add_argument('--batch_size', type=int,  default=16)
    parser.add_argument('--accum_num',  type=int,  default=1)
    parser.add_argument('--optimizer',  type=str,  default='adam')
    parser.add_argument('--lr',         type=float,default=2e-4)
    parser.add_argument('--lr_sched',   type=str,  default='constant')
    parser.add_argument('--lrf_min',    type=float,default=0.01)
    parser.add_argument('--lr_warmup',  type=int,  default=0)
    parser.add_argument('--grad_clip',  type=float,default=2.0)
    # training iterations setting
    parser.add_argument('--iterations', type=int,  default=800_000)
    parser.add_argument('--eval_first', action=argparse.BooleanOptionalAction, default=False)
    # exponential moving averaging (EMA)
    parser.add_argument('--ema',        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ema_decay',  type=float,default=0.9999)
    parser.add_argument('--ema_warmup', type=int,  default=10_000)
    # device setting
    parser.add_argument('--fixseed',    action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--workers',    type=int,  default=6)
    cfg = parser.parse_args()

    # default settings
    cfg.wdecay = 0.0
    cfg.amp = False
    cfg.wandb_log_interval = 100
    cfg.model_log_interval = 1000
    cfg.model_val_interval = 1000
    return cfg


class TrainWrapper(BaseTrainingWrapper):
    def set_dataset(self):
        cfg = self.cfg

        logging.info('==== Datasets and Dataloaders ====')
        trainset = get_image_dateset(cfg.trainset, transform_cfg=cfg.transform)
        logging.info(f'Training root: {trainset.root}')
        logging.info(f'Number of training images = {len(trainset)}')
        logging.info(f'Training transform: \n{str(trainset.transform)}')

        self.make_training_loader(trainset)

    def eval_model(self, model) -> dict:
        results = image_self_evaluate(model, dataset=self.cfg.valset, progress=False)
        return results


def main():
    cfg = parse_args()
    trainer = TrainWrapper(cfg)
    trainer.main()


if __name__ == '__main__':
    main()
