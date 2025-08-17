import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,random_split 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch.optim import Adam

class Trainer:
    """
    Trainer class for PyTorch models.

    inputs:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        loss_fn (callable): Loss function (mandatory).
        lr (float): Learning rate. Default is 1e-3.
        optimizer_class (torch.optim.Optimizer): Optimizer class. Default is Adam.
        scheduler_class (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler class.
        scheduler_kwargs (dict): Keyword arguments for scheduler.
        device (torch.device): Device to run training on. Defaults to CUDA if available.
    """
    def __init__(self, model, train_loader, val_loader, loss_fn, lr=1e-3,
                 optimizer_class=optim.Adam, scheduler_class=None, scheduler_kwargs=None,
                 device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer_class(self.model.parameters(), lr=lr)

        if scheduler_class:
            self.scheduler = scheduler_class(self.optimizer, **(scheduler_kwargs or {}))
        else:
            self.scheduler = None

    def train(self, num_epochs=500, save_dir=None):
        """
        Train the model for a given number of epochs.

        Inputs:
            num_epochs (int): Number of training epochs.
            save_dir (str): Directory to save model checkpoints. If None, no saving is done.
        Returns:
            dict: Dictionary with training and validation losses per epoch.
        """
        history = {"train_loss": [],"val_loss": []}
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0.0

            for inputs in self.train_loader:
                x, y = inputs
                #print(x.shape)
                #print(y.shape)
                x = x.to(self.device)
                y = y.to(self.device)

                #outputs = self.model(x.unsqueeze(1))
                outputs = self.model(x)
                
                
                loss = self.loss_fn(outputs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            history["train_loss"].append(avg_train_loss)
            #print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")

            if self.val_loader:
                val_loss = self.validate()
                history["val_loss"].append(val_loss)
                #print(f"Epoch {epoch}/{num_epochs}, Val Loss: {val_loss:.6f}")

            if self.scheduler:
                self.scheduler.step()

            if save_dir:
                save_path = f"{save_dir}/wnet_epoch{epoch}.pth"
                torch.save(self.model.state_dict(), save_path)
        return history

    def validate(self):
        """
        Run validation loop and calculate average validation loss.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs in self.val_loader:
                x, y = inputs
                x = x.to(self.device)
                y = y.to(self.device)
                #outputs = self.model(x.unsqueeze(1))
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        return avg_val_loss
    def evaluate(self):
        """
        Evaluate the model on the validation set using RMSE and Pearson correlation.

        Returns:
            dict: Dictionary with 'rmse' and 'pearson_r' metrics.
        """
        self.model.eval()
        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                #outputs = self.model(x.unsqueeze(1))
                outputs = self.model(x)

                y_true_all.append(y.cpu().numpy())
                y_pred_all.append(outputs.cpu().numpy())

        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)

        rmse = np.sqrt(mean_squared_error(
        y_true_all.reshape(y_true_all.shape[0], -1),
        y_pred_all.reshape(y_pred_all.shape[0], -1)
        ))
        pearson_r = pearsonr(
        y_true_all.flatten(), y_pred_all.flatten())[0]

        return {"rmse": rmse, "pearson_r": pearson_r}


    
    def save_model(self, path):
        """
        Save model weights to a file.

        Inputs:
            path (str): File path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Load model weights from a file.

        Inputs:
            path (str): File path from which to load the model.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()