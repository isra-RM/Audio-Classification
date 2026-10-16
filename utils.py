import logging

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def load_and_split_data(csv_path,
                        train_frac=0.7,
                        random_state=7):
    """Load the audio file/label CSV, encode labels, and split 70/15/15 into train/val/test.

    Returns:
        train, val, test (pd.DataFrame), num_classes (int)
    """
    data_df = pd.read_csv(csv_path)

    label_encoder = LabelEncoder()
    data_df["Class"] = label_encoder.fit_transform(data_df["Class"])

    # first split off the training set, then split the remainder evenly into val/test
    train, remainder = train_test_split(
        data_df, train_size=train_frac, random_state=random_state, stratify=data_df["Class"]
    )
    val, test = train_test_split(
        remainder, train_size=0.5, random_state=random_state, stratify=remainder["Class"]
    )

    num_classes = len(label_encoder.classes_)
    logger.info("Split shapes - train: %s, val: %s, test: %s", train.shape, val.shape, test.shape)

    return train, val, test, num_classes


def plot_training_curves(metrics):
    """Plot training/validation loss and accuracy curves from a Lightning CSVLogger metrics DataFrame."""
    train_loss = metrics[['epoch', 'train_loss']].dropna()
    val_loss = metrics[['epoch', 'val_loss']].dropna()
    train_acc = metrics[['epoch', 'train_acc']].dropna()
    val_acc = metrics[['epoch', 'val_acc']].dropna()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    axs[0].plot(train_loss['epoch'], train_loss['train_loss'], label='Training Loss')
    axs[0].plot(val_loss['epoch'], val_loss['val_loss'], label='Validation Loss')
    axs[0].set_title('Training and Validation Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(train_acc['epoch'], train_acc['train_acc'] * 100, label='Training Accuracy')
    axs[1].plot(val_acc['epoch'], val_acc['val_acc'] * 100, label='Validation Accuracy')
    axs[1].set_title('Training and Validation Accuracy over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].set_ylim([0, 100])
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("training_plot.png", dpi=150, bbox_inches="tight")
    plt.close()