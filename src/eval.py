from typing import List, Optional
from pathlib import Path

import sys
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.quantization import fit

import torch

import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger

from src.utils import utils

log = utils.get_logger(__name__)


def load_checkpoint(path, device='cpu'):
    path = Path(path).expanduser()
    if path.is_dir():
        path /= 'checkpoint_last.pt'
    # dst = f'cuda:{torch.cuda.current_device()}'
    log.info(f'Loading checkpoint from {str(path)}')
    state_dict = torch.load(path, map_location=device)
    # T2T-ViT checkpoint is nested in the key 'state_dict_ema'
    if state_dict.keys() == {'state_dict_ema'}:
        state_dict = state_dict['state_dict_ema']
    adjusted_state_dict = {}
    for key, value in state_dict['module'].items():
        print(f"Key: {key.replace('model.', '')} -> Value: {value.shape}")
        new_key = key.replace('model.', '')
        adjusted_state_dict[new_key] = value
    for key, value in adjusted_state_dict.items():
        print(f"Key: {key.replace('model.', '')} -> Value: {value.dtype}")
    return adjusted_state_dict



def evaluate(config: DictConfig) -> None:
    """Example of inference with trained model.
    It loads trained image classification model from checkpoint.
    Then it loads example image and predicts its label.
    """

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it

    # We want to add fields to config so need to call OmegaConf.set_struct
    OmegaConf.set_struct(config, False)

    # load Lightning model
    checkpoint_type = config.eval.get('checkpoint_type', 'lightning')
    if checkpoint_type not in ['lightning', 'pytorch']:
        raise NotImplementedError(f'checkpoint_type ${checkpoint_type} not supported')

    if checkpoint_type == 'lightning':
        cls = hydra.utils.get_class(config.task._target_)
        model = cls.load_from_checkpoint(checkpoint_path=config.eval.ckpt)
    else:
        model: LightningModule = hydra.utils.instantiate(config.task, cfg=config,
                                                                 _recursive_=False)
        load_return = model.model.load_state_dict(load_checkpoint(config.eval.ckpt,
                                                                          device=model.device),
                                                          strict=False)
        log.info(load_return)

    # datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule: LightningDataModule = model._datamodule
    datamodule.prepare_data()
    datamodule.setup()

    # print model hyperparameters
    log.info(f'Model hyperparameters: {model.hparams}')

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger,  _convert_="partial"
    )
    
    from neural_compressor.quantization import fit as fit
    from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion


    def eval_func_for_nc(model_n, trainer_n):
        setattr(model, "model", model_n)
        result = trainer_n.validate(model=model, dataloaders=datamodule.val_dataloader())
        print("resulttttt:", result, type(result))
        return result[0]["val/loss"]


    def eval_func(model):
        return eval_func_for_nc(model, trainer)

    conf = PostTrainingQuantConfig(approach="static", op_type_dict={
        "Embedding": {
            "weight": {
                "dtype": ["fp16"]
            },
            "activation": {
                 "dtype": ["fp16"]
            }
        }, 
        "Embedding": {
            "weight": {
                "dtype": ["fp16"]
            },
            "activation": {
                 "dtype": ["fp16"]
            }
        }
    }, excluded_precisions = ["fp16"]
    ,recipes={"smooth_quant": True})
    q_model = fit(model=model, conf=conf, calib_dataloader=datamodule.val_dataloader())
    print(sum(p.numel() for p in q_model.model.parameters()), sum(p.numel() for p in q_model.model.parameters()), "sizzesss")


    #woq_conf = PostTrainingQuantConfig(approach="dynamic")

    #quantized_model = fit(model=model, conf=woq_conf, calib_dataloader=datamodule.test_dataloader())

    # Evaluate the model
    log.info("Starting evaluation!")
    #if config.eval.get('run_val', True):
    #    trainer.validate(model=q_model.model, datamodule=datamodule)
    if config.eval.get('run_test', True):
        trainer.test(model=q_model.model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )
