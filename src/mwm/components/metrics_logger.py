import numpy as np
import mlflow
from mwm.components.metrics import *


class MetricsLogger2Channel:
    def __init__(self):
        self.epoch_loss = 0.0
        self.epoch_loss_val = 0.0
        self.epoch_iou_ch0 = 0.0
        self.epoch_iou_ch1 = 0.0

    def update_sum(self, loss, preds, masks):
        self.epoch_loss += loss.item()
        self.epoch_iou_ch0 += np.sum(iou_list(preds[:,0,:,:], masks[:,0,:,:]))
        self.epoch_iou_ch1 += np.sum(iou_list(preds[:,1,:,:], masks[:,1,:,:]))

    def update_mean(self, num_batch, num_samples):
        self.epoch_loss /= num_batch
        self.epoch_iou_ch0 /= num_samples
        self.epoch_iou_ch1 /= num_samples

    def update_sum_val(self, loss):
        self.epoch_loss_val += loss.item()
    
    def update_mean_val(self, num_batch):
        self.epoch_loss_val /= num_batch
    
    def log_metrics_mlflow(self, epoch):
        mlflow.log_metric("loss_train", self.epoch_loss, step=epoch)
        mlflow.log_metric("loss_val", self.epoch_loss_val, step=epoch)
        mlflow.log_metric("iou_ch0", self.epoch_iou_ch0, step=epoch)
        mlflow.log_metric("iou_ch1", self.epoch_iou_ch1, step=epoch)

        print(f"Epoch {epoch}, \
                Loss: {self.epoch_loss:.4f}, \
                Val Loss: {self.epoch_loss_val:.4f}, \
                Avg IoU Ch0: {self.epoch_iou_ch0:.4f}, \
                Avg IoU Ch1: {self.epoch_iou_ch1:.4f}")
        
        self.reset()
    
    def reset(self):
        self.epoch_loss = 0.0
        self.epoch_loss_val = 0.0
        self.epoch_iou_ch0 = 0.0
        self.epoch_iou_ch1 = 0.0
