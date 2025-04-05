from pathlib import Path
import torch

from mwm.utils.common import load_json
from mwm.components.model_architecture import make_model
from mwm.components.dataset import make_dataset
from mwm.components.image_processing import (
    post_processing_watershed_2ch, 
    post_processing_denoise_2ch
)

def inference(config_path:str, image_path:str):
    """
    Inference function for the segmentation model.
    Args:
        - config_path (str): Path to the configuration file. Example config see "app/config.json".
        - image_path (str): Path to the input image for segmentation.
    Returns:
        - mask_pred_uint8 (np.ndarray): Mask-level prediction in uint8 format.
        - labels_pred (np.ndarray): Label-level prediction with shape: (h, w).
    """
    # Load config
    config = load_json(Path(config_path))

    # Load model
    model = make_model(config.network, encoder_weights=None)
    model.load_state_dict(torch.load(config.model_path, map_location=torch.device("cpu")))
    model.eval()

    # Define pre/post-processing on input image
    test_dataset = make_dataset(
        config.dataset, 
        image_dir="", # root path
        mask_dir="",
        sdm_dir=None, 
        image_list=[image_path],
        mode="test",
        image_size=config.image_size
    )

    # Get input image (e.g. image shape: torch.Size([3, 3, 3, 256, 256]))
    image, _, _ = test_dataset[0]  
    
    # Handle device & batching
    _, _, c, h, w = image.shape
    image = image.reshape(-1, c, h, w).to(torch.device("cpu")) # torch.Size([9, 3, 256, 256]) 

    # Get prediction
    with torch.no_grad():
        output = model(image).squeeze()

    # Move to CPU and to numpy
    output = output.cpu().numpy() # <class 'numpy.ndarray'>, shape: (9, 2, 256, 256)

    # Mask-level result
    # <class 'numpy.ndarray'>, probabilities, 2 channels, cut to original image size
    output_stitched = test_dataset.transform.reconstruct_full_frame(output)
    output_stitched = post_processing_denoise_2ch(output_stitched)

    # Label-level result (final)
    labels_pred = post_processing_watershed_2ch(output_stitched)

    return labels_pred, output_stitched
