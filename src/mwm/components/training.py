import os
from tqdm import tqdm
import psutil
from datetime import datetime
from mwm.constants import *
from mwm.utils.common import read_yaml, load_json
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
        self.params = load_json(params_filepath)

        # Make model
        self.model = make_model(self.params.network, self.params.encoder_weights)

        # Make dataset
        self.image_dir = os.path.join(self.config.data_ingestion.unzip_dir, self.config.dataset.image_dir)
        self.mask_dir = os.path.join(self.config.data_ingestion.unzip_dir, self.config.dataset.mask_dir)
        self.image_size = self.params.image_size_lut[self.params.network]

        # TODO: update with cross-validation        
        with open(os.path.join(self.config.data_ingestion.unzip_dir, self.config.dataset.training_set_file), "r") as f:
            self.image_list_train = f.read().splitlines()[:self.params.num_training_samples]

        self.train_dataset = make_dataset(self.params.dataset, self.image_dir, self.mask_dir, self.image_list_train, self.image_size)


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
                bce_weight=self.params.weighted_dice_bce_2ch.bce_weight
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

        batch_progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.this_epoch+1}/{self.params.epochs}", leave=True)

        for images, masks in batch_progress_bar:
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()  # Reset gradients
            outputs = self.model(images)  # Forward pass
            loss = self.criterion(outputs, masks)  # Compute loss

            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update weights

            self.metrics_logger.update_sum(loss, outputs, masks)

            # Get CPU & RAM usage for display/monitoring
            ram_used = psutil.virtual_memory().used / 1024**3

            batch_progress_bar.set_postfix(loss=loss.item(), ram_used=f"{ram_used:.2f} GB", cpu_usage=f"{psutil.cpu_percent()}%")

        self.metrics_logger.update_mean(len(self.train_loader), len(self.train_dataset))
        self.metrics_logger.log_metrics_mlflow(self.this_epoch+1) # Logger is reset afterwards


    def train(self, save_model=False, save_interval=10):

        # Initialize metrics logger
        if self.params.metrics_logger == "metrics_logger_2ch":
            self.metrics_logger = MetricsLogger2Channel()
        else:
            logger.error(f"Invalid metrics logger: {self.params.metrics_logger}")
            raise ValueError(f"Invalid metrics logger: {self.params.metrics_logger}")
        
        # Define train_loader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.params.batch_size, shuffle=True)

        # Set model to training mode
        self.model.train()

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
        save_path = os.path.join(self.config.model.model_dir, f"model_epoch{self.this_epoch+1}_{timestamp}.pth")
        torch.save(self.model.state_dict(), save_path)  # Save model weights

        mlflow.log_param(f"model_epoch{self.this_epoch+1}_path", save_path)

        logger.info(f"Model saved successfully! Location: {save_path}")
    