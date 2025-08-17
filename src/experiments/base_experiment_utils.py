import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(".."))
# Assumed imports
from src.data.loading import loading_and_filtering_
from src.data.preprocessing import Preprocessor
from src.models.wnet1d import WNet1D
from src.models.loss_func import CombinedSignalLoss
from src.training.trainer import Trainer
from src.data.SignalAugmentor import SignalAugmentor

def split_dataset(dataset, train_ratio=0.8):
    '''
    Splits a dataset into training and validation subsets based on a specified ratio.

    This split is performed sequentially, using the first portion of the data for training
    and the remaining for validation. This is particularly useful for time-series data
    (like biosignals), where shuffling could introduce information leakage.

    Parameters:
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to be split. Assumes indexing yields (x, y) pairs.
    
    train_ratio : float, optional (default=0.8)
        Proportion of the dataset to include in the training split. Must be between 0 and 1.

    Returns:
    -------
    train_subset : torch.utils.data.Subset
        The training subset (first `train_ratio` portion of the dataset).
    
    val_subset : torch.utils.data.Subset
        The validation subset (remaining portion of the dataset).
    '''
    train_size = int(train_ratio * len(dataset))
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def get_dataloaders(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def apply_augmentation(dataset, augmentor, augmentation_type):
    """
    Applies a specific augmentation method to each (x, y) sample in the dataset.

    This function supports augmentation methods that return either:
    - a single tensor (e.g., x_augmented), or
    - a tuple (x_augmented, y_augmented)

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset yielding (x, y) samples.

    augmentor : SignalAugmentor
        An instance containing augmentation methods like jitter, mixup, etc.

    augmentation_type : str
        The name of the augmentation method to apply. Must be a method of `augmentor`.

    Returns
    -------
    list
        A list of (x_aug, y_aug) tuples ready to be wrapped into a Dataset.
    """
    augmented_samples = []

    for x, y in dataset:
        method = getattr(augmentor, augmentation_type, None)
        result = method(x, y)  # some methods may ignore y or return single tensor
        if isinstance(result, tuple) and len(result) == 2:
            x_aug, y_aug = result
        else:
            x_aug, y_aug = result, y  # assume only x is augmented, y is unchanged

        augmented_samples.append((x_aug, y_aug))

    return augmented_samples
    

def plot_and_save_loss_curves(history, record, results_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curve for Record {record}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    figure_name = f"base_model_{record.replace('/', '_')}.png"
    figure_path = os.path.join(results_dir, figure_name)
    plt.savefig(figure_path)
    print(f"Saved training plot to {figure_path}")
    plt.show()

def train_on_record(record, batch_size, learning_rate, num_epochs, loss_fn, device, results_dir, augmentation_type=None,jitter_std=None,permute_segments=None,crop_ratio=None,
                    mixup_alpha=None,cutmix_alpha=None):
    """
    Trains and evaluates a WNet1D model on a single patient record with optional data augmentation.

    The function performs the following steps:
    1. Loads and preprocesses the dataset for the given record.
    2. Splits the dataset into training and validation sets (last 20% for validation to prevent leakage).
    3. Optionally applies data augmentation on the training set and appends augmented samples.
    4. Creates DataLoaders for training and validation.
    5. Initializes and trains the model.
    6. Evaluates performance (RMSE and Pearson correlation) on the validation set.
    7. Saves the trained model and loss curves to the specified results directory.

    Parameters
    ----------
    record : str
        Path to the patient record (e.g., "record_01").

    batch_size : int
        Number of samples per batch used in training and validation.

    learning_rate : float
        Learning rate for the optimizer.

    num_epochs : int
        Number of epochs to train the model.

    loss_fn : callable
        Custom loss function used during training (e.g., CombinedSignalLoss()).

    device : torch.device
        The device to run the model on (e.g., torch.device("cuda") or "cpu").

    results_dir : str
        Directory path where model checkpoints and training plots are saved.

    augmentation_type : str, optional
        Name of the augmentation method to apply from the SignalAugmentor class (e.g., "jitter", "mixup").
        If None, no augmentation is applied.

    Returns
    -------
    metrics : dict
        Dictionary containing evaluation metrics on the validation set. Includes:
            - "rmse": Root Mean Square Error
            - "pearson_r": Pearson correlation coefficient
    """
    dataset = Preprocessor(record)
    train_dataset, val_dataset = split_dataset(dataset)

    # Apply augmentation only on training data
    if augmentation_type:
        augmentor = SignalAugmentor(
            jitter_std=jitter_std,
            permute_segments=permute_segments,
            crop_ratio=crop_ratio,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
        )
        if augmentation_type=='multiple_methods':#run multiple all methods that yiled posetive and stable results(cutmix, permutation and jitter)
            augmented_samples_cutmix = apply_augmentation(train_dataset, augmentor, "cutmix")
            augmented_samples_permutation = apply_augmentation(train_dataset, augmentor, "permutation")
            augmented_samples_jitter = apply_augmentation(train_dataset, augmentor, "jitter")
            all_augmented_samples = (augmented_samples_cutmix +augmented_samples_permutation +augmented_samples_jitter)
            # Wrap them into a TensorDataset (since they are already tensors)
            augmented_dataset = torch.utils.data.TensorDataset(
                torch.stack([x for x, _ in all_augmented_samples]),
                torch.stack([y for _, y in all_augmented_samples])
            )

            # Combine with the original dataset
            train_dataset = ConcatDataset([train_dataset, augmented_dataset])
            
            
        else:
            augmented_samples = apply_augmentation(train_dataset, augmentor, augmentation_type)
            train_dataset = ConcatDataset([
                train_dataset,
                torch.utils.data.TensorDataset(
                    torch.stack([x for x, _ in augmented_samples]),
                    torch.stack([y for _, y in augmented_samples])
                )
            ])

    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size)

    model = WNet1D().to(device)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        lr=learning_rate,
        optimizer_class=Adam,
        scheduler_class=StepLR,
        scheduler_kwargs={"step_size": 800, "gamma": 0.1},
        device=device
    )

    history = trainer.train(num_epochs=num_epochs)
    metrics = trainer.evaluate()

    print(f"Record {record} --> RMSE: {metrics['rmse']:.4f}, Pearson r: {metrics['pearson_r']:.4f}")

    model_name = f"base_model_{record.replace('/', '_')}.pt"
    model_path = os.path.join(results_dir, model_name)
    trainer.save_model(model_path)
    print(f"Saved model to {model_path}")

    plot_and_save_loss_curves(history, record, results_dir)

    return metrics

def run_experiment(
    database_path="../bidmc",
    records_file_path="../bidmc/RECORDS",
    required_signal=["PLETH,", "II,"],
    sampling_freq=125,
    min_duration=8,
    batch_size=128,
    learning_rate=1e-3,
    num_epochs=500,
    results_dir="../results/base_model",
    augmentation_type=None,
    jitter_std=0.02,
    permute_segments=6,
    crop_ratio=0.9,
    mixup_alpha=0.2,
    cutmix_alpha=0.2):
    os.makedirs(results_dir, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_records = loading_and_filtering_(
        database_path=database_path,
        file_path=records_file_path,
        min_duration=min_duration,
        required_signal=required_signal,
        sampling_freq=sampling_freq
    )

    loss_fn = CombinedSignalLoss()
    rmse_list, pearson_list = [], []

    for record in valid_records:
        print(f"Processing Record: {record}")
        metrics = train_on_record(
            record=record,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            loss_fn=loss_fn,
            device=device,
            results_dir=results_dir,
            augmentation_type=augmentation_type,
            jitter_std=jitter_std,
            permute_segments=permute_segments,
            crop_ratio=crop_ratio,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha
        )

        rmse_list.append(metrics["rmse"])
        pearson_list.append(metrics["pearson_r"])

    mean_rmse = np.mean(rmse_list)
    mean_pearson = np.mean(pearson_list)

    np.save(os.path.join(results_dir, "rmse_list.npy"), rmse_list)
    np.save(os.path.join(results_dir, "pearson_list.npy"), pearson_list)

    print(f"Mean RMSE: {mean_rmse:.4f}")
    print(f"Mean Pearson r: {mean_pearson:.4f}")

    # Plot histograms
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(rmse_list, bins=15, color='skyblue', edgecolor='black')
    plt.axvline(mean_rmse, color='red', linestyle='--', label=f'Mean = {mean_rmse:.4f}')
    plt.title("RMSE Histogram")
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(pearson_list, bins=15, color='lightgreen', edgecolor='black')
    plt.axvline(mean_pearson, color='red', linestyle='--', label=f'Mean = {mean_pearson:.4f}')
    plt.title("Pearson Correlation Histogram")
    plt.xlabel("Pearson r")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "RMSE_and_r.png")
    plt.savefig(save_path, dpi=300)
    plt.show()


