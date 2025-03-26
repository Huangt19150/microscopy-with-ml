from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from skimage import measure
import mlflow

from mwm import logger
from mwm.constants import *
from mwm.utils.common import read_yaml, load_json
from mwm.config.configuration import get_params
from mwm.components.model_architecture import *
from mwm.components.dataset import *
from mwm.components.image_processing import (
    read_image_png, 
    post_processing_watershed_2ch, 
    post_processing_denoise_2ch
)
from mwm.components.metrics import iou_object_labels, measures_at


class EvaluationProcessor2Channel:
    def __init__(self):
        self.results = []
        self.thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)


    def prep_evaluation(self, prediction, mask_path):
        """
        Args:
            - prediction: after denoise operations
            - mask_path: path to the original
        """

        # Mask to label
        self.sample_name = os.path.basename(mask_path).split(".")[0]
        mask_raw = read_image_png(mask_path)
        self.labels_gt = measure.label(mask_raw[:,:,0], background=0)

        # Prediction to label
        self.labels_pred = post_processing_watershed_2ch(prediction) # key post-processing logic
            

    def update_metrics(self):
        iou_matrix = iou_object_labels(self.labels_gt, self.labels_pred)
        if iou_matrix.size == 0:
            mean_object_iou = 0.0
        else:
            mean_object_iou = np.max(iou_matrix, axis=0).mean()
        
        # Calculate F1 score at all thresholds
        for t in self.thresholds:
            f1, precision, recall, jaccard, tp, fp, fn = measures_at(t, iou_matrix)
            res = {
                "Sample": self.sample_name, 
                "Threshold": t, 
                "F1": f1, 
                "Precision": precision, 
                "Recall": recall, 
                "Jaccard": jaccard, 
                "MeanObjectIoU": mean_object_iou,
                "TP": tp, 
                "FP": fp, 
                "FN": fn
                }
            self.results.append(res)
    

    def log_key_metrics_to_mlflow(self):
        df = pd.DataFrame(self.results)
        df_agg = df.drop(columns=["Sample"]).groupby("Threshold").mean().reset_index().sort_values("Threshold", ascending=True)
        df_agg_list = df_agg.to_dict("records")
        for row_dict in df_agg_list:
            metrics = {k: v for k, v in row_dict.items() if k != "Threshold"}
            mlflow.log_metrics(metrics, step=int(row_dict["Threshold"]*100))
        mlflow.log_metric("MAF1", df_agg["F1"].mean())
        mlflow.log_metric("MAPrecision", df_agg["Precision"].mean())
        mlflow.log_metric("MARecall", df_agg["Recall"].mean())
        mlflow.log_metric("MAJaccard", df_agg["Jaccard"].mean())
        mlflow.log_param("thresholds", self.thresholds)


    def save_results(self, output_path):
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)


class Evaluator():
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = get_params(params_filepath, "evaluation")

        # Make & load model
        self.model_path = self.params.model_file_path
        if not os.path.exists(self.model_path):
            logger.info("Please enter a valid model file path in params.json.\nPath can be found in mlfow trainning logs.\nExample: 'artifacts/models/model_epoch1_20250224_112303.pth'")
            logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        elif not self.model_path.endswith(".pth"):
            logger.error(f"Check if model file ends with '.pth'. Invalid model file: {self.model_path}")
            raise ValueError(f"Invalid model file: {self.model_path}")
        
        self.model = make_model(self.params.network, encoder_weights=None)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device("cpu")))
        logger.info(f"Model loaded from: {self.model_path}")

        # Make dataset
        self.image_dir = os.path.join(self.config.data_ingestion.unzip_dir, self.config.dataset.image_dir)
        self.mask_dir = os.path.join(self.config.data_ingestion.unzip_dir, self.config.dataset.mask_dir)

        with open(os.path.join(self.config.data_ingestion.unzip_dir, self.config.dataset.test_set_file), "r") as f:
            self.image_list_test = f.read().splitlines()
        self.test_dataset = make_dataset(
            self.params.dataset, 
            self.image_dir, 
            self.mask_dir,
            None, 
            self.image_list_test,
            "test",
            self.params.image_size
        )

        # Make save path (optional)
        if self.params.save_predictions:
            model_name = os.path.basename(self.model_path).split(".")[0]
            self.save_dir = os.path.join(self.config.evaluation.evaluation_dir, f"{model_name}_predictions")
            os.makedirs(self.save_dir, exist_ok=True)


    def handle_device(self):
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)


    def evaluate(self):
        self.evaluate_processor = EvaluationProcessor2Channel()
        # Set model to evaluation mode
        self.model.eval()

        # Evaluate individual sample without batching
        batch_progress_bar = tqdm(self.test_dataset, desc=f"Evaluation", leave=True)
        with torch.no_grad():
            for image, _, _ in batch_progress_bar: # e.g. image shape: torch.Size([3, 3, 3, 256, 256])

                mask_path = self.test_dataset.get_mask_path()
                
                # Handle device & batching
                _, _, c, h, w = image.shape
                image = image.reshape(-1, c, h, w).to(self.device) # torch.Size([9, 3, 256, 256]) 

                # Get prediction
                output = self.model(image).squeeze()

                # Move to CPU and to numpy
                output = output.cpu().numpy() # <class 'numpy.ndarray'>, shape: (9, 2, 256, 256)

                # <class 'numpy.ndarray'>, probabilities, 2 channels, cut to original image size
                output_stitched = self.test_dataset.transform.reconstruct_full_frame(output)
                output_stitched = post_processing_denoise_2ch(output_stitched)

                # Evaluate
                self.evaluate_processor.prep_evaluation(output_stitched, mask_path)
                self.evaluate_processor.update_metrics()

                if self.params.save_predictions:
                    save_path = os.path.join(self.save_dir, os.path.basename(mask_path))

                    empty_channel = np.zeros_like(output_stitched[:,:,0])
                    mask_pred_uint8 = np.stack([output_stitched[:,:,1], empty_channel, output_stitched[:,:,0]], axis=-1) * 255 # cv2 uses BGR
                    cv2.imwrite(save_path, mask_pred_uint8)

        mlflow.set_experiment("Evaluation")
        with mlflow.start_run():

            self.evaluate_processor.log_key_metrics_to_mlflow()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.config.evaluation.evaluation_dir,
                f"evaluation_{timestamp}_on_{os.path.basename(self.model_path).split('.')[0]}.csv"
            )
            self.evaluate_processor.save_results(save_path)

            mlflow.log_param("evaluation_save_path", save_path)
            mlflow.log_param("model_path", self.model_path)
    