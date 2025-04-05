import os
from pathlib import Path
from PIL import Image
import gradio as gr

from mwm.utils.common import load_json
from mwm.components.inference import inference
from mwm.components.visualization import (
    colorize_label_map,
    colorize_mask_2d
)

def segment(image: Image.Image):

    # Dump image
    # TODO: add normalization
    image_savepath = "input_image.png"
    image.save(image_savepath)


    labels_pred, mask_pred = inference(
        config_path="config.json",
        image_path=image_savepath
    )

    mask_pred_colored = colorize_mask_2d(mask_pred) # TODO: show later
    labels_pred_colored = colorize_label_map(labels_pred)

    # Clean up
    os.remove(image_savepath)

    return Image.fromarray(labels_pred_colored)

# TODO: remove after local testing
working_dir = "app"
os.chdir(working_dir)

# Load config
config = load_json(Path("config.json"))

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
