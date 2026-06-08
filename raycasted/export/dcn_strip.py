"""RayCastED — DCN Stripping for ONNX Export.

Replaces C3k2_DCN and DCNConv modules with standard equivalents
so the model can be exported to ONNX/TensorRT.
"""

import torch.nn as nn
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Conv


def _strip_dcn_conv(module: nn.Module) -> nn.Module:
    """Replace DCNConv with equivalent Conv + BN + SiLU."""
    from raycasted.model.blocks.dcn import DCNConv

    if not isinstance(module, DCNConv):
        return module

    c1 = module.dcn.weight.shape[1]
    c2 = module.dcn.weight.shape[0]
    k = module.kernel_size
    replacement = Conv(c1, c2, k)
    replacement.conv.weight.data.copy_(module.dcn.weight.data)
    replacement.bn = module.bn
    replacement.act = module.act
    return replacement


def _strip_dcn_bottleneck(module: nn.Module) -> nn.Module:
    """Replace DCNBottleneck with standard Bottleneck."""
    from raycasted.model.blocks.dcn_blocks import DCNBottleneck

    if not isinstance(module, DCNBottleneck):
        return module

    c1 = module.cv1.dcn.weight.shape[1]
    c2 = module.cv1.dcn.weight.shape[0]
    replacement = Bottleneck(c1, c2, shortcut=module.add, g=1, e=1.0)
    replacement.cv1 = _strip_dcn_conv(module.cv1)
    replacement.cv2.conv.weight.data.copy_(module.cv2.conv.weight.data)
    if module.cv2.conv.bias is not None:
        replacement.cv2.conv.bias.data.copy_(module.cv2.conv.bias.data)
    replacement.cv2.bn = module.cv2.bn
    return replacement


def strip_dcn_from_model(model: nn.Module) -> nn.Module:
    """Replace all C3k2_DCN modules with standard C3k2.

    The resulting model can be exported to ONNX without custom ops.
    """
    from ultralytics.nn.modules.block import C3k2

    from raycasted.model.blocks.dcn_blocks import C3k2_DCN

    for name, child in list(model.named_children()):
        if isinstance(child, C3k2_DCN):
            c = child.c
            n = len(child.m)
            c1 = child.cv1.conv.weight.shape[0]
            c2 = child.cv2.conv.weight.shape[0]
            e = c / c2

            replacement = C3k2(c1, c2, n=n, c3k=False, e=e, shortcut=True, g=1)
            replacement.cv1 = child.cv1
            replacement.cv2 = child.cv2
            for i, sub in enumerate(child.m):
                replacement.m[i] = _strip_dcn_bottleneck(sub)

            setattr(model, name, replacement)

        elif isinstance(child, (nn.Sequential, nn.ModuleList)):
            for i, sub in enumerate(child):
                if isinstance(sub, C3k2_DCN):
                    c = sub.c
                    n = len(sub.m)
                    c1 = sub.cv1.conv.weight.shape[0]
                    c2 = sub.cv2.conv.weight.shape[0]
                    e = c / c2

                    replacement = C3k2(c1, c2, n=n, c3k=False, e=e, shortcut=True, g=1)
                    replacement.cv1 = sub.cv1
                    replacement.cv2 = sub.cv2
                    for j, item in enumerate(sub.m):
                        replacement.m[j] = _strip_dcn_bottleneck(item)

                    child[i] = replacement
                else:
                    strip_dcn_from_model(sub)
        elif not isinstance(child, (nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity)):
            strip_dcn_from_model(child)

    return model
