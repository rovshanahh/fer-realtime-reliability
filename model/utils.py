import torch.nn as nn


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def set_finetune_resnet50(
    model,
    *,
    train_fc: bool = True,
    unfreeze_layer4: bool = False,
    unfreeze_layer3: bool = False,
) -> None:
    """
    For torchvision resnet50:
      - layer3: second-to-last residual stage
      - layer4: last residual stage
      - fc: classifier head

    Typical strategies:
      frozen (fc only): train_fc=True,  unfreeze_layer3=False, unfreeze_layer4=False
      layer4+fc:        train_fc=True,  unfreeze_layer3=False, unfreeze_layer4=True
      layer3+4+fc:      train_fc=True,  unfreeze_layer3=True,  unfreeze_layer4=True
      full:             (don't call this helper; just don't freeze anything)
    """
    freeze_all(model)

    if train_fc:
        if not hasattr(model, "backbone") or not hasattr(model.backbone, "fc") or model.backbone.fc is None:
            raise RuntimeError("Expected model.backbone.fc to exist, but it doesn't.")
        unfreeze_module(model.backbone.fc)

    if unfreeze_layer3:
        if not hasattr(model, "backbone") or not hasattr(model.backbone, "layer3") or model.backbone.layer3 is None:
            raise RuntimeError("Expected model.backbone.layer3 to exist, but it doesn't.")
        unfreeze_module(model.backbone.layer3)

    if unfreeze_layer4:
        if not hasattr(model, "backbone") or not hasattr(model.backbone, "layer4") or model.backbone.layer4 is None:
            raise RuntimeError("Expected model.backbone.layer4 to exist, but it doesn't.")
        unfreeze_module(model.backbone.layer4)