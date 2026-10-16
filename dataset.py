from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_spectrogram(file_path, spec_dir="./spectrograms"):
    """Audio file path -> z-scored spectrogram array, read from the saved PNG."""
    # same layout as SpectrogramGenerator: <spec_dir>/<parent folder name>/<stem>.png
    img_path = Path(spec_dir).resolve() / Path(file_path).parent.name / f"{Path(file_path).stem}.png"
    spec = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    # per-sample z-score, same as the original get_spectrogram
    return (spec - spec.mean()) / (spec.std() + 1e-6)


class CustomAudioDataset(Dataset):
    def __init__(self, dataframe, spec_dir="./spectrograms"):
        self.dataframe = dataframe.reset_index(drop=True)
        self.labels = torch.tensor(list(self.dataframe["Class"]), dtype=torch.long)
        self.audios = [
            torch.tensor(load_spectrogram(p, spec_dir), dtype=torch.float32)
            for p in self.dataframe["FilePath"]
        ]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio = self.audios[idx].unsqueeze(0)  # (1, H, W)
        return audio, self.labels[idx]