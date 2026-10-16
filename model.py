from pathlib import Path
import timm
import torchmetrics
import torch
import torch.nn as nn
import pytorch_lightning as pl

WEIGHTS = Path(__file__).resolve().parent / "model" / "resnet18" / "model.safetensors"


class AudioClassifier(pl.LightningModule):
    def __init__(self, 
                 num_classes, 
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["csv_path", "dataset_root"])

        self.model = timm.create_model(
            'resnet18',
            pretrained=True,
            pretrained_cfg_overlay=dict(file=str(WEIGHTS)),  # local file, no download
            in_chans=1,
            num_classes=num_classes,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        backbone_params = [p for n, p in self.model.named_parameters() if not n.startswith('fc.')]
        head_params = [p for n, p in self.model.named_parameters() if n.startswith('fc.')]

        # low LR for pretrained ResNet18 layers and higher LR for the freshly-initialized classification head
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': head_params, 'lr': 1e-3},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}