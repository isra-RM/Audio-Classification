import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import CustomAudioDataset

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, spec_dir="./spectrograms", batch_size=16, num_workers=0):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.spec_dir = spec_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = CustomAudioDataset(self.train_df, self.spec_dir)
        self.val_dataset = CustomAudioDataset(self.val_df, self.spec_dir)
        self.test_dataset = CustomAudioDataset(self.test_df, self.spec_dir)

        self.persistent_workers = self.num_workers > 0  # Keep workers alive between epochs for faster data loading

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)