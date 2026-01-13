import os
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from matplotlib import pyplot as plt
import torch
import numpy as np
from xlstm_mixer.lit.enums import ForecastingTaskOptions
from lightning.pytorch.loggers import WandbLogger
import wandb


class ForecastVisualizeCallback(Callback):

    def __init__(self, task_options: ForecastingTaskOptions = ForecastingTaskOptions.MULTIVARIATE_2_MULTIVARIATE, pred_len: int= 0, idxs: range = range(0,4), freq_epoch: int = 4 ) -> None:

        self.task_options = task_options
        self.pred_len = pred_len
        self.idxs = idxs
        self.freq_epoch = freq_epoch

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.freq_epoch != 0:
            return

        # make grid of predictions/ground truth
        batch_x = []
        batch_y = []
        batch_x_mark = []
        batch_y_mark = []
        
        # Check if dataset is large enough
        idxs = [i for i in self.idxs if i < len(trainer.datamodule.train_dataset)]
        if not idxs:
            return

        for i in idxs:
            elem_x,elem_y,elem_x_mark, elem_y_mark = trainer.datamodule.train_dataset[i]

            batch_x.append(elem_x)
            batch_y.append(elem_y)
            batch_x_mark.append(elem_x_mark)
            batch_y_mark.append(elem_y_mark)
        
        batch_x = torch.from_numpy(np.stack(batch_x)).float().to(pl_module.device)
        batch_y = torch.from_numpy(np.stack(batch_y)).float().to(pl_module.device)
        batch_x_mark = torch.from_numpy(np.stack(batch_x_mark)).float().to(pl_module.device)
        batch_y_mark = torch.from_numpy(np.stack(batch_y_mark)).float().to(pl_module.device)
        
        pl_module.eval()
        with torch.no_grad():
            dec_input = None
            outputs = pl_module.model(batch_x, batch_x_mark, dec_input, batch_y_mark)

            f_dim = -1 #-1 if self.task_options == ForecastingTaskOptions.MULTIVARIATE_2_UNIVARIATE else 0
            outputs = outputs[:, -self.pred_len:, f_dim:].contiguous().detach().cpu().numpy()
            batch_y = batch_y[:, -self.pred_len:, f_dim:].contiguous().detach().cpu().numpy()
            
            # Plotting
            num_plots = len(idxs)
            cols = 2
            rows = (num_plots + 1) // 2
            
            fig, axs = plt.subplots(rows, cols, figsize=(10, 5 * rows))
            if num_plots == 1:
                axs = [axs]
            
            axs_flat = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

            for i, ax in enumerate(axs_flat):
                if i < num_plots:
                    # Select the last channel to plot if multivariate
                    # Shape is (seq_len, channels) or (channels, seq_len) - assuming (seq_len, channels) based on code
                    # Usually time series are (seq_len, channels).
                    # Let's plot the last channel which is often the target in univariate, or one of them in multivariate.
                    
                    # If shape is (batch, seq, dim)
                    # outputs[i] is (seq, dim)
                    
                    # Plot last channel
                    ax.plot(outputs[i, :, -1], label='pred')
                    ax.plot(batch_y[i, :, -1], label='true')
                    ax.legend()
                    ax.set_title(f"Sample {i}")
                else:
                    ax.axis('off')
            
            plt.tight_layout()

            # Save to WandB
            if isinstance(trainer.logger, WandbLogger):
                trainer.logger.experiment.log({"forecast_prediction": wandb.Image(fig)})
            
            # Save locally
            os.makedirs('pics', exist_ok=True)
            plt.savefig(f'pics/epoch_{trainer.current_epoch}.pdf', bbox_inches='tight')
            plt.close(fig)

        pl_module.train()
