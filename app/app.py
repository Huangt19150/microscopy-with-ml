import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import gradio as gr

from mwm.utils.common import load_json
from mwm.components.model_architecture import make_model
from mwm.components.dataset import make_dataset
from mwm.components.image_processing import (
    post_processing_watershed_2ch, 
    post_processing_denoise_2ch
)

# TODO: remove after local testing
working_dir = "app"
os.chdir(working_dir)

# Load config
config = load_json(Path("config.json"))

# Load model
model = make_model(config.network, encoder_weights=None)
model.load_state_dict(torch.load(config.model_path, map_location=torch.device("cpu")))
model.eval()


def colorize_label_map(label_map):
    unique_labels = np.unique(label_map)
    n = unique_labels.max() + 1

    # Random colormap: shape (n_labels, 3), dtype=uint8
    colormap = np.random.randint(0, 255, size=(n, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]  # make background black if label 0

    # Create color image
    h, w = label_map.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for label in unique_labels:
        color_image[label_map == label] = colormap[label]

    return color_image


def segment(image: Image.Image):

    # Dump image
    # TODO: add normalization
    image_savepath = "input_image.png"
    image.save(image_savepath)

    # Define pre/post-processing on input image
    test_dataset = make_dataset(
        config.dataset, 
        image_dir="", # root path
        mask_dir="",
        sdm_dir=None, 
        image_list=[image_savepath],
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

    # <class 'numpy.ndarray'>, probabilities, 2 channels, cut to original image size
    output_stitched = test_dataset.transform.reconstruct_full_frame(output)
    output_stitched = post_processing_denoise_2ch(output_stitched)

    # Mask-level result
    # TODO: show later
    empty_channel = np.zeros_like(output_stitched[:,:,0])
    mask_pred_uint8 = np.stack([output_stitched[:,:,1], empty_channel, output_stitched[:,:,0]], axis=-1) * 255

    # Label-level result (final)
    labels_pred = post_processing_watershed_2ch(output_stitched)
    colored = colorize_label_map(labels_pred)

    # Clear up
    os.remove(image_savepath)

    return Image.fromarray(colored)

# Gradio Interface
demo = gr.Interface(
    fn=segment,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="PyTorch Image Segmentation Demo",
    description="Upload an image or try one of the examples below to run nuclei segmentation.",
    examples=config.example_images
)

if __name__ == "__main__":
    demo.launch()
