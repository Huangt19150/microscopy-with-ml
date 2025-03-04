import segmentation_models_pytorch as smp
from mwm import logger

# Utils for model architecture
def make_model(network_name):
    if network_name == "unet_resnet34_2ch":
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2)
        logger.info(f"Model: {network_name} successfully created. ")
        return model
    else:
        logger.error(f"Invalid network: {network_name}")
        raise ValueError(f"Invalid network: {network_name}")


# Classes for customized model architecture