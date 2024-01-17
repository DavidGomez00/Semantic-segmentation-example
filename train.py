import random

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler

import config
from dataset import CarvanaDatamodule
from model import UNET_pl


def main():
    ############################# Reproducibility ############################
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    ##########################################################################

    ############################# Experiment #####################################################

    # To make lightning happy
    torch.set_float32_matmul_precision('medium')

    # Logger and profiler
    logger = TensorBoardLogger("tb_logs", "U-Net")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/U-Net"),
        trace_memory = True,
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20)
    )
    # Model and datamodule
    model = UNET_pl(in_channels=3, out_channels=1)
    datamodule = CarvanaDatamodule(data_dir=config.DATA_DIR,
                                   image_height=config.IMAGE_HEIGHT,
                                   image_width=config.IMAGE_WIDTH,
                                   batch_size=config.BATCH_SIZE,
                                   num_workers=config.NUM_WORKERS
                                   )
    # Trainer
    trainer = L.Trainer(accelerator=config.ACCELERATOR,
                        devices=config.DEVICES,
                        min_epochs=config.MIN_EPOCHS,
                        max_epochs=config.NUM_EPOCHS,
                        precision=config.PRECISION,
                        logger=logger,
                        callbacks=[
                            EarlyStopping(monitor="train_loss", min_delta=0.001, patience=4),
                            ModelCheckpoint(
                                dirpath="model_ckpt/",
                                filename='U-Net_'+'{epoch}-{val_loss:.2f}',
                                monitor="val_loss",
                                save_weights_only=True)],
                        log_every_n_steps=16,                   
                        )
    trainer.fit(model,datamodule)
    ################################################################################################


if __name__=="__main__":
    main()