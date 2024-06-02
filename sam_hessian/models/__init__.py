from .resnet import *
from .wideresnet import *

def get_model(model_name, num_classes):
    if model_name == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif model_name == "resnet34":
        return ResNet34(num_classes=num_classes)
    elif model_name == "resnet50":
        return ResNet50(num_classes=num_classes)
    elif model_name == "resnet101":
        return ResNet101(num_classes=num_classes)
    elif model_name == "resnet152":
        return ResNet152(num_classes=num_classes)
    elif model_name == "wrn28_10":
        return WideResnet28_10(num_classes=num_classes)
    else:
        raise ValueError("Invalid model!!!")