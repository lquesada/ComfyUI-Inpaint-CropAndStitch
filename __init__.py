from .inpaint_cropandstitch import InpaintCrop
from .inpaint_cropandstitch import InpaintStitch
from .inpaint_cropandstitch import InpaintExtendOutpaint

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "InpaintCrop": InpaintCrop,
    "InpaintStitch": InpaintStitch,
    "InpaintExtendOutpaint": InpaintExtendOutpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCrop": "✂️ Inpaint Crop",
    "InpaintStitch": "✂️ Inpaint Stitch",
    "InpaintExtendOutpaint": "✂️ Extend Image for Outpainting",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
