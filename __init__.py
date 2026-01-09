from .inpaint_cropandstitch import InpaintCropImproved
from .inpaint_cropandstitch import InpaintStitchImproved

WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "InpaintCropImproved": InpaintCropImproved,
    "InpaintStitchImproved": InpaintStitchImproved,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCropImproved": "✂️ Inpaint Crop",
    "InpaintStitchImproved": "✂️ Inpaint Stitch",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
