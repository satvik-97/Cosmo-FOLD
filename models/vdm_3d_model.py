import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import autograd, Tensor
from torch.nn.functional import mse_loss
from typing import Optional, Tuple
from torch.special import expm1
from tqdm import trange
from torch.distributions.normal import Normal
from lightning.pytorch import LightningModule
from transformers import get_cosine_schedule_with_warmup
from utils.utils import FixedLinearSchedule, LearnedLinearSchedule, NNSchedule, kl_std_normal
from timm.scheduler import CosineLRScheduler


class VDM(nn.Module):
    def __init__(
        self,
        score_model: nn.Module,
        noise_schedule: str = "fixed_linear",
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        antithetic_time_sampling: bool = True,
        image_shape: Tuple[int] = (
            1,
            64,
            64,
            64,     #######################
        ),
        data_noise: float = 1.0e-3,
        lambdas: Tuple[float] = (1.0, 1.0, 1.0),
    ):
        """Variational diffusion model, continuous time implementation of arxiv:2107.00630.

        Args:
            score_model (nn.Module): model used to denoise
            noise_schedule (str, optional): whether fixed_linear or learned noise schedules.
            Defaults to "fixed_linear".
            gamma_min (float, optional): minimum gamma value. Defaults to -13.3.
            gamma_max (float, optional): maximum gamma value. Defaults to 5.0.
            antithetic_time_sampling (bool, optional): whether to do antithetic time sampling.
            Defaults to True.
            image_shape (Tuple[int], optional): image shape. Defaults to ( 3, 32, 32, ).
            data_noise (float, optional): noise in data, used for reconstruction loss.
            Defaults to 1.0e-3.

        Raises:
            ValueError: when noise_schedule not in (fixed_linear, learned_linear)
        """
        super().__init__()
        self.score_model = score_model
        self.image_shape = image_shape
        self.data_noise = data_noise
        if noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(gamma_min, gamma_max)
        elif noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(gamma_min, gamma_max)
        elif noise_schedule == "learned_nn":
            self.gamma = NNSchedule(gamma_min, gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {noise_schedule}")
        self.antithetic_time_sampling = antithetic_time_sampling
        self.lambdas = lambdas

    def variance_preserving_map(
        self, x: Tensor, times: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """Add noise to data sample, in a variance preserving way Eq. 10 in arxiv:2107.00630

        Args:
            x (Tensor): data sample
            times (Tensor): time steps
            noise (Tensor, optional): noise to add. Defaults to None.

        Returns:
            Tensor: Noisy sample
        """
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            times = times.view((times.shape[0],) + (1,) * (x.ndim-1))
            gamma_t = self.gamma(times)
        alpha = torch.sqrt(torch.sigmoid(-gamma_t))
        scale = torch.sqrt(torch.sigmoid(gamma_t))
        if noise is None:
            noise = torch.randn_like(x)
        return alpha * x + noise * scale, gamma_t

    def sample_times(
        self,
        batch_size: int,
        device: str,
    ) -> Tensor:
        """Sample diffusion times for batch, used for monte carlo estimates

        Args:
            batch_size (int): size of batch

        Returns:
            Tensor: times
        """
        if self.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=device)
        else:
            times = torch.rand(batch_size, device=device)
        return times

    def get_diffusion_loss(
        self,
        gamma_t: Tensor,
        times: Tensor,
        pred_noise: Tensor,
        noise: Tensor,
        bpd_factor: float,
    ) -> float:
        """get loss for diffusion process. Eq. 17 in arxiv:2107.00630

        Args:
            gamma_t (Tensor): gamma at time t
            times (Tensor): time steps
            pred_noise (Tensor): noise prediction
            noise (Tensor): noise added

        Returns:
            float: diffusion loss
        """
        gamma_grad = autograd.grad(  # gamma_grad shape: (B, )
            gamma_t,  # (B, )
            times,  # (B, )
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        pred_loss = (
            # ((pred_noise - noise) ** 2).flatten(start_dim=1).sum(axis=-1)
            ((pred_noise - noise)**2).flatten(start_dim=1).sum(axis=-1)
        )  # (B, )
        return bpd_factor * 0.5 * pred_loss * gamma_grad

    def get_latent_loss(
        self,
        x: Tensor,
        bpd_factor: float,
    ) -> float:
        """Latent loss to ensure the prior is truly Gaussian

        Args:
            x (Tensor): data sample

        Returns:
            float: latent loss
        """
        gamma_1 = self.gamma(torch.tensor([1.0], device=x.device))
        sigma_1_sq = torch.sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * x**2
        return bpd_factor * kl_std_normal(mean_sq, sigma_1_sq).flatten(start_dim=1).sum(
            axis=-1
        )
    
    def get_reconstruction_loss(
        self,
        x: Tensor,
        bpd_factor: float,
    ):
        """Measure reconstruction error

        Args:
            x (Tensor): data sample

        Returns:
            float: reconstruction loss
        """
        noise_0 = torch.randn_like(x)
        times = torch.tensor([0.0], device=x.device)
        z_0, gamma_0 = self.variance_preserving_map(
            x,
            times=times,
            noise=noise_0,
        )
        # Generate a sample for z_0 -> closest to the data
        alpha_0 = torch.sqrt(torch.sigmoid(-gamma_0))
        z_0_rescaled = z_0 / alpha_0
        # assert not torch.isnan(z_0).any(), "z_0 contains NaNs"
        # assert not torch.isnan(gamma_0).any(), "gamma_0 contains NaNs"
        ############################### 
        #EXTRA Quantile loss
        # tau = 0.9
        # error = z_0_rescaled - x 
        # quantile_loss = torch.where(error >= 0, tau * error, (tau - 1) * error)
        # quantile_loss = quantile_loss.flatten(start_dim=1).sum(axis=-1)
        # print( -bpd_factor * (Normal(loc=z_0_rescaled, scale=self.data_noise).log_prob(x).flatten(start_dim=1).sum(axis=-1)),bpd_factor *quantile_loss*32*32*32)
        return -bpd_factor * (Normal(loc=z_0_rescaled, scale=self.data_noise).log_prob(x).flatten(start_dim=1).sum(axis=-1))
    
    def get_loss(
        self,
        x: Tensor,
        conditioning: Optional[Tensor] = None,
        noise: Optional[Tensor] = None
    ) -> float:
        """Get loss for diffusion model. Eq. 11 in arxiv:2107.00630

        Args:
            x (Tensor): data sample
            conditioning (Optional[Tensor], optional): conditioning. Defaults to None.
            noise (Optional[Tensor], optional): noise. Defaults to None.

        Returns:
            float: loss
        """
        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2))
        # Sample from q(x_t | x_0) with random t.
        times = self.sample_times(
            x.shape[0],
            device=x.device,
        ).requires_grad_(True)
        if noise is None:
            noise = torch.randn_like(x)
        x_t, gamma_t = self.variance_preserving_map(x=x, times=times, noise=noise)
        # Predict noise added
        pred_noise = self.score_model(
            x_t,
            conditioning=conditioning,
            g_t=gamma_t.squeeze(),
        )

        # *** Diffusion loss
        diffusion_loss = self.get_diffusion_loss(
            gamma_t=gamma_t,
            times=times,
            pred_noise=pred_noise,
            noise=noise,
            bpd_factor=bpd_factor,
        )*(self.lambdas[0] if self.lambdas is not None else 1)

        # *** Latent loss: KL divergence from N(0, 1) to q(z_1 | x)
        latent_loss = self.get_latent_loss(
            x=x,
            bpd_factor=bpd_factor,
        )*(self.lambdas[1] if self.lambdas is not None else 1)
        
        # *** Reconstruction loss:  - E_{q(z_0 | x)} [log p(x | z_0)].
        recons_loss = self.get_reconstruction_loss(
            x=x,
            bpd_factor=bpd_factor,
        )*(self.lambdas[2] if self.lambdas is not None else 1)


        # *** Overall loss, Shape (B, ).
        loss = diffusion_loss + latent_loss + recons_loss

        metrics = {
            "elbo": loss.mean(),
            "diffusion_loss": diffusion_loss.mean(),
            "latent_loss": latent_loss.mean(),
            "reconstruction_loss": recons_loss.mean(),
        }
        return loss.mean(), metrics

    def alpha(self, gamma_t: Tensor) -> Tensor:
        """Eq. 4 arxiv:2107.00630

        Args:
            gamma_t (Tensor): gamma evaluated at t

        Returns:
            Tensor: alpha
        """
        return torch.sqrt(torch.sigmoid(-gamma_t))

    def sigma(self, gamma_t):
        """Eq. 3 arxiv:2107.00630

        Args:
            gamma_t (Tensor): gamma evaluated at t

        Returns:
            Tensor: sigma
        """
        return torch.sqrt(torch.sigmoid(gamma_t))

    def sample_zs_given_zt(
        self,
        zt: Tensor,
        conditioning: Tensor,
        t: Tensor,
        s: Tensor,
    ) -> Tensor:
        """Sample p(z_s|z_t, x) used for standard ancestral sampling. Eq. 34 in arxiv:2107.00630

        Args:
            z (Tensor): latent variable at time t
            conditioning (Tensor): conditioning for samples
            t (Tensor): time t
            s (Tensor): time s

        Returns:
            zs, samples for time s
        """
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        c = -expm1(gamma_s - gamma_t)
        alpha_t = self.alpha(gamma_t)
        alpha_s = self.alpha(gamma_s)
        sigma_t = self.sigma(gamma_t)
        sigma_s = self.sigma(gamma_s)
        pred_noise = self.score_model(
            zt,
            conditioning=conditioning,
            g_t=gamma_t,
        )
        mean = alpha_s / alpha_t * (zt - c * sigma_t * pred_noise)
        scale = sigma_s * torch.sqrt(c)
        return mean + scale * torch.randn_like(zt)

    @torch.no_grad()
    def sample(
        self,
        conditioning: Tensor,
        batch_size: int,
        n_sampling_steps: int,
        device: str,
        z: Optional[Tensor] = None,
        return_all=False,
        verbose=False
    ) -> Tensor:
        """Generate new samples given some conditioning vector

        Args:
            conditioning (Tensor): conditioning
            batch_size (int): number of samples in batch
            n_sampling_steps (int): number of sampling steps
            device (str, optional): device to run model. Defaults to "cpu".
            z (Optional[Tensor], optional): initial latent variable. Defaults to None.

        Returns:
            Tensor: generated sample
        """
        if z is None:
            z = torch.randn(
                (batch_size, *self.image_shape),
                device=device,
            )
        steps = torch.linspace(
            1.0,
            0.0,
            n_sampling_steps + 1,
            device=device,
        )
        if return_all:
            zs=[]
        for i in (trange(n_sampling_steps, desc="sampling") if verbose else range(n_sampling_steps)):
            z = self.sample_zs_given_zt(
                zt=z,
                conditioning=conditioning,
                t=steps[i],
                s=steps[i + 1],
            )
            if return_all:
                zs.append(z)
        if return_all:
            return torch.stack(zs,dim=0)
        return z


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, decay=1.0):
        super().__init__(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch)
        self.decay = decay
        # store original lrs
        self.initial_lrs = list(self.base_lrs)

    def step(self, epoch=None):
        # If stepping epoch-by-epoch
        if epoch is None:
            # auto mode: when next step would cross boundary
            if self.T_cur + 1 == self.T_i:
                # shrink peaks for next cycle
                self.base_lrs = [lr * self.decay for lr in self.base_lrs]
        else:
            # stepped with epoch argument
            # find which cycle you’re in (n) and adjust base_lrs accordingly
            # similar logic to parent implementation but with decay
            import math
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0
            self.base_lrs = [init_lr * (self.decay**n) for init_lr in self.initial_lrs]

        super().step(epoch)



class LightVDM(LightningModule):
    def __init__(
        self,
        score_model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1.0e-6,
        n_sampling_steps: int = 100,
        image_shape: Tuple[int] = (1, 64, 64, 64),
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        #gamma_min_max: float = -9.0,
        #gamma_max_min: float = 4.0,
        draw_figure=None,
        dataset='illustris',
        **kwargs
    ):
        """Variational diffusion wrapper for lightning

        Args:
            score_model (nn.Module): model used to denoise
            learning_rate (float, optional): initial learning rate. Defaults to 1.0e-4.
            n_sampling_steps (int, optional): number of steps used to sample validations.
            Defaults to 250.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["score_model","draw_figure"])
        self.val_losses = []  # Initialize the validation loss container
        # self.val_pred_sqsum = torch.zeros(3)
        # self.val_true_sqsum = torch.zeros(3)
        # self.val_count = 0
        self.model = VDM(
            score_model=score_model,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            #gamma_min_max=gamma_min_max,
            #gamma_max_min=gamma_max_min,
            image_shape=image_shape,
            **kwargs
        )
        self.dataset=dataset
        # print("VDM:", self.dataset)
        self.draw_figure=draw_figure
        if self.draw_figure is None:
            def draw_figure(args,**kwargs):
                fig=plt.figure(figsize=(5,5))
                return fig
            self.draw_figure=draw_figure

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        """get loss for samples

        Args:
            x (Tensor): data samples
            conditioning (Tensor): conditioning

        Returns:
            Tensor: loss
        """
        return self.model.get_loss(x=x, conditioning=conditioning)

    def evaluate(self, batch: Tuple, stage: str = None) -> Tensor:
        """get loss function

        Args:
            batch (Tuple): batch of examples
            stage (str, optional): training stage. Defaults to None.

        Returns:
            Tensor: loss function
        """
        conditioning, x = batch
        if conditioning.ndim == 2:
            conditioning = None
        loss, metrics = self(x=x, conditioning=conditioning)
        if self.logger is not None:
            self.logger.log_metrics(metrics)
            
        # if return_pred:
        #     return loss, y_pred_R, y_true_R

        return loss

    def training_step(
        self,
        batch: Tuple,
        batch_idx: int,
    ) -> Tensor:
        """Training

        Args:
            batch (Tuple): batch of examples
            batch_idx (int): batch idx

        Returns:
            Tensor: loss
        """
        return self.evaluate(batch, "train")

    def draw_samples(
        self,
        conditioning: Tensor,
        batch_size: int,
        n_sampling_steps: int,
        z: Optional[Tensor] = None,  # Add z as an optional argument
        verbose=False,
        return_all=False
    ) -> Tensor:
        """draw samples from model

        Args:
            conditioning (Tensor): conditioning
            batch_size (int): number of samples in batch
            n_sampling_steps (int): number of sampling steps used to generate validation
            samples

        Returns:
            Tensor: generated samples
        """
        return self.model.sample(
            conditioning=conditioning,
            batch_size=batch_size,
            n_sampling_steps=n_sampling_steps,
            z=z,  # Pass z into the sample function
            device=conditioning.device,
            verbose=verbose,
            return_all=return_all
        )

    # def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
    #     """validate model

    #     Args:
    #         batch (Tuple): batch of examples
    #         batch_idx (int): idx for batch

    #     Returns:
    #         Tensor: loss
    #     """
    #     # print("LENGTH OF BATCH",(batch.shape))
    #     conditioning, x = batch    
    #     loss = 0    
        
    #     if batch_idx == 0:
    #         sample = self.draw_samples(
    #             conditioning=conditioning,
    #             batch_size=len(x),
    #             n_sampling_steps=self.hparams.n_sampling_steps,
    #         )
    #         loss = mse_loss(x, sample)
    #         # fig = self.draw_figure(x,sample,conditioning,self.dataset)
    #         self.log_dict({'val_loss': loss}, on_epoch=True)
            
    #         # if self.logger is not None:
    #         #     self.logger.experiment.log_figure(figure=fig)
    #         # plt.close()
            
    #     return loss
    




    # def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
    #     """validate model

    #     Args:
    #         batch (Tuple): batch of examples
    #         batch_idx (int): idx for batch

    #     Returns:
    #         Tensor: loss
    #     """
    #     # print("LENGTH OF BATCH",(batch.shape))
    #     # conditioning, x = batch    
    #     loss = 0    
        
    #     if batch_idx == 0:
    #         loss = self.evaluate(batch, "valid")
    #         self.log_dict({'val_loss': loss}, on_epoch=True)
    #         return loss
    
    
    
    
    # @torch.no_grad()
    # def _accum_std_metrics(self, batch):
    #     # accept (cond, y_R, *_) OR dict
    #     if isinstance(batch, (list, tuple)):
    #         cond, y_true_R = batch[0], batch[1]
    #     else:
    #         cond, y_true_R = batch["cond"], batch["y_R"]       # arcsinh targets [B,3,...]
    #     y_pred_R = self.model.score_model(cond)                         # predicted arcsinh residuals
    #     B = y_true_R.shape[0]
    #     device = y_true_R.device

    #     # s  = self.s_b.to(device)
    #     # mu = self.mu_b.to(device)
    #     # sg = self.sig_b.to(device)

    #     # def bshape(t): return t.view(1, -1, *([1]*(y_true_R.ndim-2)))
    #     y_pred_X = torch.sinh(y_pred_R)
    #     y_true_X = torch.sinh(y_true_R)

    #     y_pred_X = y_pred_X.reshape(B, 3, -1)
    #     y_true_X = y_true_X.reshape(B, 3, -1)

    #     self.val_pred_sqsum += (y_pred_X**2).sum(dim=(0,2)).cpu()
    #     self.val_true_sqsum += (y_true_X**2).sum(dim=(0,2)).cpu()
    #     self.val_count += B * y_pred_X.shape[-1]

    def validation_step(self, batch: Tuple, batch_idx: int) -> dict:
        # Compute the loss for the current batch
        loss = self.evaluate(batch, "valid")
        # Append loss to the instance list (for aggregation)
        self.val_losses.append(loss)
        # Return the loss dict (if needed)
        # self._accum_std_metrics(batch)  #New
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        # Aggregate the losses from all validation batches
        avg_loss = torch.stack(self.val_losses).mean()
        # Log only the averaged loss for the epoch (visible in Comet)
        self.log("val_loss", avg_loss, prog_bar=True, on_epoch=True)
        # Clear the list for the next epoch
        self.val_losses.clear()

        # NEW: compute and log std-ratio per band
        # if self.val_count > 0:
        #     std_pred = (self.val_pred_sqsum / self.val_count).sqrt()
        #     std_true = (self.val_true_sqsum / self.val_count).sqrt()
        #     ratio = (std_pred / (std_true + 1e-12)).tolist()
        #     for i, r in enumerate(ratio):
        #         self.log(f"val_std_ratio_band{i}", r, prog_bar=False, on_epoch=True)

        # # reset accumulators
        # self.val_pred_sqsum.zero_()
        # self.val_true_sqsum.zero_()
        # self.val_count = 0
            


    def test_step(self, batch, batch_idx):
        """test model

        Args:
            batch (Tuple): batch of examples
            batch_idx (int): idx for batch

        Returns:
            Tensor: loss
        """
        return self.evaluate(batch, "test")



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=20,
            eta_min=5e-6

        )

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(
    #         self.parameters(),
    #         lr=self.hparams.learning_rate,
    #         weight_decay=self.hparams.weight_decay,
    #     )

    #     scheduler = CosineAnnealingWarmRestartsDecay(
    #     optimizer,
    #     T_0=15,          # your first cycle length
    #     T_mult=2,       # if you want fixed-length cycles
    #     eta_min=self.hparams.learning_rate / 50,
    #     decay=0.6,
    # )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # "interval": "epoch",   # or "step" if you prefer
            },
        }
