import os
import sys

# NOTE: Deploy ONLY: Add the src directory to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

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
    # TODO: add normalization before allow user to upload their own image
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


# Load config
config = load_json(Path("config.json"))

# Gradio Interface
demo = gr.Interface(
    fn=segment,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="🔬 Microscopy Image Segmentation with Machine Learning",
    description="""
    ## Work-in-progress project exploring state-of-the-art (SOTA) ML models for nuclei (cell body) segmentation in microscopy images. 
    Check out full details in this repo: [microscopy-with-ml](https://github.com/Huangt19150/microscopy-with-ml)

    ## 📋 Quick Start Guide:
    1. Try one of the **examples** below to run nuclei segmentation.
    2. (Upload your own image is not yet supported. Coming soon!)

    ## 📖 Reference:
    1. Dataset: [BBBC039v1](https://bbbc.broadinstitute.org/BBBC039) Caicedo et al. 2018, available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012](https://www.nature.com/articles/nmeth.2083).
    2. Method improved from [topcoders](https://www.kaggle.com/competitions/data-science-bowl-2018/discussion/54741)
    
    """,
    examples=config.example_images,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
