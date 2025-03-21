import os
from tqdm import tqdm
import psutil
from datetime import datetime
from mwm.constants import *
from mwm.utils.common import read_yaml, load_json
from mwm.config.configuration import get_params
from mwm import logger

# Model architecture
import segmentation_models_pytorch as smp
from mwm.components.model_architecture import *

# Dataset
from mwm.components.dataset import *
from torch.utils.data import DataLoader

# Loss
from mwm.components.loss import *

# Metrics logger
import mlflow
from mwm.components.metrics_logger import *


class Training:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = get_params(params_filepath, "training")

        # Make model
        self.model = make_model(self.params.network, self.params.encoder_weights)

        # Make dataset
        self.image_dir = os.path.join(self.config.data_ingestion.unzip_dir, self.config.dataset.image_dir)
        self.mask_dir = os.path.join(self.config.data_ingestion.unzip_dir, self.config.dataset.mask_dir)
        self.sdm_dir = os.path.join(self.config.data_ingestion.unzip_dir, self.config.dataset.sdm_dir)

        # TODO: update with cross-validation
        # - Train dataset
        with open(os.path.join(self.config.data_ingestion.unzip_dir, self.config.dataset.training_set_file), "r") as f:
            self.image_list_train = f.read().splitlines()[:self.params.num_training_samples]

        self.train_dataset = make_dataset(
            self.params.dataset, 
            self.image_dir, 
            self.mask_dir,
            self.sdm_dir,
            self.image_list_train, 
            "train",
            self.params.image_size
        )

        # - Validation dataset
        with open(os.path.join(self.config.data_ingestion.unzip_dir, self.config.dataset.validation_set_file), "r") as f:
            self.image_list_val = f.read().splitlines()
        
        self.val_dataset = make_dataset(
            self.params.dataset,
            self.image_dir,
            self.mask_dir,
            self.sdm_dir,
            self.image_list_val,
            "val",
            self.params.image_size
        )


    def handle_device(self):
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    

    def make_criterion(self):
        if self.params.loss == "weighted_dice_bce_2ch":
            self.criterion = WeightedDiceBCELoss(
                weight_1=self.params.weighted_dice_bce_2ch.weight_1,
                weight_2=self.params.weighted_dice_bce_2ch.weight_2,
                weight_3=self.params.weighted_dice_bce_2ch.weight_3,
                bce_weight=self.params.weighted_dice_bce_2ch.bce_weight,
                use_focal=self.params.weighted_dice_bce_2ch.use_focal,
                use_gradient_loss=self.params.weighted_dice_bce_2ch.use_gradient_loss,
                use_dist_loss=self.params.weighted_dice_bce_2ch.use_dist_loss
            )
            logger.info(f"Loss: {self.params.loss} selected. ")
        else:
            logger.error(f"Invalid loss: {self.params.loss}")
            raise ValueError(f"Invalid loss: {self.params.loss}")
        

    def make_optimizer(self):
        if self.params.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
            logger.info(f"Optimizer: {self.params.optimizer} selected. ")
        else:
            logger.error(f"Invalid optimizer: {self.params.optimizer}")
            raise ValueError(f"Invalid optimizer: {self.params.optimizer}")
    

    def train_epoch(self):

        batch_progress_bar = tqdm(range(self.params.steps_per_epoch), desc=f"Epoch {self.this_epoch}/{self.params.epochs-1}", leave=True)

        ### Training Phase ###
        self.model.train()

        for step in batch_progress_bar:
            images, masks, sdms = next(iter(self.train_loader))
            images, masks, sdms = images.to(self.device), masks.to(self.device), sdms.to(self.device)

            self.optimizer.zero_grad()  # Reset gradients
            outputs = self.model(images)  # Forward pass
            loss = self.criterion(outputs, masks, sdms)  # Compute loss

            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update weights

            self.metrics_logger.update_sum(loss, outputs.cpu(), masks.cpu())

            # Get CPU & RAM usage for display/monitoring
            ram_used = psutil.virtual_memory().used / 1024**3

            batch_progress_bar.set_postfix(loss=loss.item(), ram_used=f"{ram_used:.2f} GB", cpu_usage=f"{psutil.cpu_percent()}%")

        self.metrics_logger.update_mean(
            self.params.steps_per_epoch,
            self.params.steps_per_epoch * self.params.batch_size
        )

        ### Validation Phase ###
        self.model.eval()

        batch_progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.this_epoch}/{self.params.epochs-1} validation", leave=True)

        with torch.no_grad():
            for images, masks, sdms in batch_progress_bar:
                images, masks, sdms = images.to(self.device), masks.to(self.device), sdms.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks, sdms)

                self.metrics_logger.update_sum_val(loss)

                batch_progress_bar.set_postfix(val_loss=loss.item())
        
        self.metrics_logger.update_mean_val(len(self.val_loader))

        self.metrics_logger.log_metrics_mlflow(self.this_epoch) # Logger is reset afterwards


    def train(self, save_model=False, save_interval=10):

        # Initialize metrics logger
        if self.params.metrics_logger == "metrics_logger_2ch":
            self.metrics_logger = MetricsLogger2Channel()
        else:
            logger.error(f"Invalid metrics logger: {self.params.metrics_logger}")
            raise ValueError(f"Invalid metrics logger: {self.params.metrics_logger}")
        
        # Define data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.params.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.params.batch_size, shuffle=False)

        # Start training
        mlflow.set_experiment("Training")
        with mlflow.start_run():
            for epoch in range(self.params.epochs):
                self.this_epoch = epoch
                self.train_epoch()

                if save_model:
                    if (epoch+1) % save_interval == 0:
                        self.save_model()
            
            mlflow.log_params(self.params.to_dict())
        
        logger.info("Training completed. ")


    def save_model(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.config.model.model_dir, f"model_epoch{self.this_epoch}_{timestamp}.pth")
        torch.save(self.model.state_dict(), save_path)  # Save model weights

        mlflow.log_param(f"model_epoch{self.this_epoch}_path", save_path)

        logger.info(f"Model saved successfully! Location: {save_path}")
    