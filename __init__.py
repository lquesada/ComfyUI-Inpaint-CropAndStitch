from .inpaint_cropandstitch import InpaintCrop
from .inpaint_cropandstitch import InpaintStitch
from .inpaint_cropandstitch import InpaintExtendOutpaint
from .inpaint_cropandstitch import InpaintResize

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "InpaintCrop": InpaintCrop,
    "InpaintStitch": InpaintStitch,
    "InpaintExtendOutpaint": InpaintExtendOutpaint,
    "InpaintResize": InpaintResize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCrop": "✂️ Inpaint Crop",
    "InpaintStitch": "✂️ Inpaint Stitch",
    "InpaintExtendOutpaint": "✂️ Extend Image for Outpainting",
    "InpaintResize": "✂️ Resize Image Before Inpainting",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
