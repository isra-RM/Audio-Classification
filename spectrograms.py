import argparse
import logging
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    handlers = [logging.StreamHandler()]

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

setup_logging()

class SpectrogramGenerator:
    def __init__(self, 
                 out_dir="./spectrograms", 
                 sr=22050, duration=5,
                 hop_length=512, 
                 n_fft=2048, 
                 img_height=128, 
                 img_width=256):
        
        self.out_dir = Path(out_dir).resolve()
        self.sr = sr
        self.duration = duration
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.img_height = img_height
        self.img_width = img_width

    def get_spectrogram(self, file_path):
        """Audio file -> mel spectrogram in dB, resized to (img_height, img_width)."""
        signal, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
        logger.debug("Loaded %s (%d samples, sr=%d)", file_path, len(signal), sr)
        spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.img_height
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)

        spec_db = librosa.util.fix_length(spec_db, size=self.duration * sr // self.hop_length + 1)
        spec_db = resize(spec_db, (self.img_height, self.img_width), anti_aliasing=True)

        mean, std = spec_db.mean(), spec_db.std()
        norm_spec_db = ((spec_db - mean) / (std + 1e-6)).astype(np.float32)

        return norm_spec_db

    def save_spectrogram(self, file_path):
        spec = self.get_spectrogram(file_path)
        # min-max scale to 0-255 so it can be stored as an 8-bit grayscale PNG
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        img = Image.fromarray((spec * 255).astype(np.uint8), mode="L")
        # keep the audio file's parent folder name as a subfolder
        name_dir = Path(file_path).parent.name
        out_path = self.out_dir / name_dir / f"{Path(file_path).stem}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return out_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="/home/neuro/code/Audio-Classification/files_paths.csv", help="CSV with a 'FilePath' column")
    parser.add_argument("--out_dir", default="./spectrograms", help="Folder to save the spectrograms")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    generator = SpectrogramGenerator(out_dir=args.out_dir)

    failed = []
    for i, file_path in enumerate(df["FilePath"], 1):
        try:
            generator.save_spectrogram(file_path)
        except Exception:
            logger.exception("Failed to process %s", file_path)
            failed.append(file_path)
        if i % 500 == 0 or i == len(df):
            logger.info("Files processed: %d out of %d", i, len(df))

    logger.info("Done! %d succeeded, %d failed.", len(df) - len(failed), len(failed))