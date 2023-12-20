import sys

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

__all__ = ["STN"]


class ConvGroupDepthwise(torch.nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size, **kwargs):
        super().__init__(
            self.depthwise_separable_conv(in_c, out_c, kernel_size, **kwargs),
            torch.nn.BatchNorm3d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d((2, 2, 2)),
        )

    def depthwise_separable_conv(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        kernels_per_layer: int = 1,
    ):
        """3D depthwise conv layer."""
        conv3d = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = torch.nn.Conv3d(
            in_channels=in_channels * kernels_per_layer,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        return torch.nn.Sequential(conv3d, pointwise_conv)


class LocalizationNet(torch.nn.Module):
    def __init__(self, input_size: tuple[int], **kwargs):
        super().__init__()

        num_kernels = int(kwargs["hparams"].stn_num_kernels)

        conv1 = ConvGroupDepthwise(
            input_size[0],
            num_kernels,
            kwargs["hparams"].stn_kernel_sizes[0],
            padding="same",
        )
        conv2 = ConvGroupDepthwise(
            num_kernels,
            num_kernels,
            kwargs["hparams"].stn_kernel_sizes[1],
            padding="same",
        )
        conv3 = ConvGroupDepthwise(
            num_kernels,
            num_kernels,
            kwargs["hparams"].stn_kernel_sizes[2],
            padding="same",
        )

        self.conv_layers = torch.nn.Sequential(conv1, conv2, conv3)
        self.polling = torch.nn.AdaptiveAvgPool3d((2, 2, 2))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(num_kernels * 2 * 2 * 2, kwargs["hparams"].stn_fc_units[0]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(kwargs["hparams"].stn_dropout),
            torch.nn.Linear(kwargs["hparams"].stn_fc_units[0], 3 * 4),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.polling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(-1, 3, 4)
        return x


class FCGroup(torch.nn.Sequential):
    def __init__(self, in_c, out_c, dropout_rate, **kwargs):
        super().__init__(
            torch.nn.Linear(in_c, out_c, bias=False),
            torch.nn.BatchNorm1d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout_rate),
        )


class STN(pl.LightningModule):
    def __init__(self, input_size: tuple[int], **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = torch.nn.MSELoss()
        self.validation_step_outputs = []
        self.test_set_outputs = []
        self.validation_logs = []

        self.localization = LocalizationNet(input_size, hparams=self.hparams)
        # initialize the weights/bias with identity transformation
        self.localization.fc[-1].weight.data.zero_()
        self.localization.fc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float)
        )

        conv = ConvGroupDepthwise
        conv1 = conv(input_size[0], 64, 5)
        conv2 = conv(64, 128, 5)
        conv3 = conv(128, 256, 5)
        self.conv_layers = torch.nn.Sequential(conv1, conv2, conv3)

        polling = torch.nn.AdaptiveAvgPool3d((2, 2, 2))
        flatten = torch.nn.Flatten()
        self.flatten = (
            torch.nn.Sequential(polling, flatten)
            if self.hparams.adaptive_pooling
            else flatten
        )

        self.fc1 = FCGroup(
            256 * (2**3 if self.hparams.adaptive_pooling else 3**3),
            self.hparams.num_fc_units[0],
            self.hparams.dropout,
        )

        self.linear = torch.nn.Linear(self.hparams.num_fc_units[-1], 1)

    def forward(self, x):
        # stn localization
        theta = self.localization(x)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.linear(x)
        return x

    @staticmethod
    def add_specific_args(parent_parser):
        """Add model specific arguments to the parser; accessible with self.hparams."""
        # fmt: off
        parser = parent_parser.add_argument_group("Model args")
        parser.add_argument("--optim", type=str, default="Adam")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--eps", type=float, default=1e-8)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--wdecay", type=float, default=0.0)
        parser.add_argument("--num-fc-units", type=int, nargs="+", default=[1000], help="Number of neurons in each fc layer")
        # parser.add_argument("--num-fc-layers", type=int, default=1, help="Number of fc layers (not useful if `--fc-layers` is already specified; can be used to specify the number of fc layers in a hyperparameter search).")
        # parser.add_argument("--num-conv-layers", type=int, default=1, help="Number of conv layers.")
        # parser.add_argument("--num-kernels", type=int, nargs="+", default=[16], help="Number of kernels in each conv layer.")
        # parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[3], help="Kernel size in each conv layer.")
        parser.add_argument("--depthwise-convs", action="store_true", help="Use depthwise separable convolutions.")
        parser.add_argument("--adaptive-pooling", action="store_true", help="Use adaptive pooling before flattening.")
        parser.add_argument("--stn-fc-units", type=int, nargs="+", default=[32], help="Number of neurons in each fc layer of the STN.")
        parser.add_argument("--stn-dropout", type=float, default=0.0, help="Dropout rate in the STN.")
        parser.add_argument("--stn-num-kernels", type=int, default=8, help="Number of kernels in each conv layer of the STN.")
        parser.add_argument("--stn-kernel-sizes", type=int, nargs="+", default=[7, 3, 1], help="Kernel size in each conv layer of the STN.")
        # fmt: on
        return parent_parser

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.optim)(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.eps,
            weight_decay=self.hparams.wdecay,
        )

    def shared_step(self, batch, batch_idx, stage):
        x, y = batch
        y_pred = self(x)

        log_params = {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
        }

        loss = self.loss_fn(y_pred, y.reshape(y.shape[0], 1))
        self.log(f"{stage}_loss", loss, **log_params)

        if stage == "train":
            pearsonr = torch.corrcoef(torch.stack((y_pred.squeeze(), y)))[0][1]
            self.log("train_pearsonr", pearsonr, **log_params)

        out = {
            f"{stage}_loss": loss,
            # for calculating metrics on validation and test set:
            "preds": y_pred,
            "labels": y,
        }

        return out

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, stage="train")
        return out["train_loss"]

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, stage="val")
        self.validation_step_outputs.append(out)
        return out["val_loss"]

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, stage="test")
        self.test_step_outputs.append(out)
        return out["test_loss"]

    def on_validation_epoch_end(self) -> None:
        out = self.validation_step_outputs
        preds = torch.cat([x["preds"] for x in out]).squeeze()
        labels = torch.cat([x["labels"] for x in out])

        pearsonr = torch.corrcoef(torch.stack((preds, labels)))[0][1]
        loss = torch.stack([x["val_loss"] for x in out]).mean()

        log = {"val_pearsonr": pearsonr, "val_loss": loss}
        self.log_dict(log, prog_bar=True, logger=True)
        self.validation_logs.append(log)

        self.validation_step_outputs.clear()

    def on_train_end(self) -> None:
        best_pearsonr = max(self.validation_logs, key=lambda x: x["val_pearsonr"])
        best_loss = min(self.validation_logs, key=lambda x: x["val_loss"])

        self.logger.experiment.track(
            {
                "best_val_pearsonr": best_pearsonr["val_pearsonr"],
                "best_val_loss": best_loss["val_loss"],
            },
            context={"subset": "val"},
        )

    def on_test_epoch_end(self) -> None:
        out = self.test_step_outputs
        preds = torch.cat([x["preds"] for x in out]).squeeze()
        labels = torch.cat([x["labels"] for x in out])

        self.log_dict(
            {
                "test_pearsonr": torch.corrcoef(torch.stack((preds, labels)))[0][1],
                "test_loss": torch.stack([x["test_loss"] for x in out]).mean(),
            },
            prog_bar=True,
            logger=True,
        )

        self.test_step_outputs.clear()
