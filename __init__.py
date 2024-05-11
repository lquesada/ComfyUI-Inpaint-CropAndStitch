from .inpaint_cropandstitch import InpaintCrop
from .inpaint_cropandstitch import InpaintStitch

NODE_CLASS_MAPPINGS = {
    "InpaintCrop": InpaintCrop,
    "InpaintStitch": InpaintStitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCrop": "✂️ Inpaint Crop",
    "InpaintStitch": "✂️ Inpaint Stitch"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
