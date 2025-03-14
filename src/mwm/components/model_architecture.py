import segmentation_models_pytorch as smp
from mwm import logger

# Utils for model architecture
def make_model(network_name, encoder_weights):
    if network_name == "unet_resnet34_2ch":
        model = smp.Unet(encoder_name="resnet34", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_efficientnet_b0_2ch":
        model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_efficientnet_b2_2ch":
        model = smp.Unet(encoder_name="efficientnet-b2", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_efficientnet_b3_2ch":
        model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    else:
        logger.error(f"Invalid network: {network_name}")
        raise ValueError(f"Invalid network: {network_name}")

    logger.info(f"Model: {network_name} successfully created. ")
    return model

# Classes for customized model architecture
