import segmentation_models_pytorch as smp
from mwm import logger

# Utils for model architecture
def freeze_encoder(model, freeze_encoder_layers):
    """
    Freeze the encoder layers of the model.
    """
    if not freeze_encoder_layers: # list is empty
        logger.info("No Encoder layer is frozen. All layers are trainable.")
    elif freeze_encoder_layers[0] == "all": # freeze all layers
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info("All Encoder layers are frozen.")
    else:
        for name, child in model.encoder.named_children():
            if name in freeze_encoder_layers:
                for param in child.parameters():
                    param.requires_grad = False
        logger.info(f"Encoder layers: {freeze_encoder_layers} are selectively frozen.")


def make_model(network_name, encoder_weights):
    if network_name == "unet_resnet34_2ch":
        model = smp.Unet(encoder_name="resnet34", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_resnet101_2ch":
        model = smp.Unet(encoder_name="resnet101", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_resnet152_2ch":
        model = smp.Unet(encoder_name="resnet152", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_efficientnet_b0_2ch":
        model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_efficientnet_b2_2ch":
        model = smp.Unet(encoder_name="efficientnet-b2", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_efficientnet_b3_2ch":
        model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_efficientnet_b4_2ch":
        model = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_dpn92_2ch":
        model = smp.Unet(encoder_name="dpn92", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_inceptionresnetv2_2ch":
        model = smp.Unet(encoder_name="inceptionresnetv2", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_regnety_040_2ch":
        model = smp.Unet(encoder_name="tu-regnety_040", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_convnext_base_2ch":
        model = smp.Unet(encoder_name="tu-convnext_base", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_convnextv2_base_2ch":
        model = smp.Unet(encoder_name="tu-convnextv2_base", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_swinv2_small_window8_256_2ch":
        model = smp.Unet(encoder_name="tu-swinv2_small_window8_256", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_edgenext_small_2ch":
        model = smp.Unet(encoder_name="tu-edgenext_small", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_edgenext_base_2ch":
        model = smp.Unet(encoder_name="tu-edgenext_base", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_efficientformerv2_s1_2ch":
        model = smp.Unet(encoder_name="tu-efficientformerv2_s1", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    elif network_name == "unet_efficientformerv2_l_2ch":
        model = smp.Unet(encoder_name="tu-efficientformerv2_l", encoder_weights=encoder_weights, in_channels=3, classes=2, activation="sigmoid")
    else:
        logger.error(f"Invalid network: {network_name}")
        raise ValueError(f"Invalid network: {network_name}")

    logger.info(f"Model: {network_name} successfully created. ")
    print("Named children in Encoder:")
    for name, _ in list(model.encoder.named_children()):
        print(name)
    return model

# Classes for customized model architecture
