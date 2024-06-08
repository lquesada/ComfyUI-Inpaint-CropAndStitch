ComfyUI-Inpaint-CropAndStitch

Copyright (c) 2024, Luis Quesada Torres - https://github.com/lquesada | www.luisquesada.com

Check ComfyUI here: https://github.com/comfyanonymous/ComfyUI

# Overview

"✂️  Inpaint Crop" is a node that crops an image before sampling. The context area can be specified via the mask, expand pixels and expand factor or via a separate (optional) mask.

"✂️  Inpaint Stitch" is a node that stitches the inpainted image back into the original image without altering unmasked areas.

"✂️  Extend Image for Outpainting" is a node that extends an image and masks in order to use the power of Inpaint Crop and Stich (rescaling, blur, blend, restitching) for outpainting.

The main advantages of inpainting only in a masked area with these nodes are:
  - It's much faster than sampling the whole image.
  - It enables setting the right amount of context from the image for the prompt to be more accurately represented in the generated picture.
  - It enables upscaling before sampling in order to generate more detail, then stitching back in the original picture.
  - It enables downscaling before sampling if the area is too large, in order to avoid artifacts such as double heads or double bodies.
  - It enables forcing a specific resolution (e.g. 1024x1024 for SDXL models).
  - It doesn't modify the unmasked part of the image, not even passing it through VAE encode and decode.
  - The nodes take care of good blending.

# Video Tutorial

[![Video Tutorial](https://img.youtube.com/vi/u5cNrumkz2w/0.jpg)](https://www.youtube.com/watch?v=u5cNrumkz2w)

[(click to open in YouTube)](https://www.youtube.com/watch?v=u5cNrumkz2w)

## Parameters
- `context_expand_pixels`: how much to grow the context area (i.e. the area for the sampling) around the original mask, in pixels. This provides more context for the sampling.
- `context_expand_factor`: how much to grow the context area (i.e. the area for the sampling) around the original mask, as a factor, e.g. 1.1 is grow 10% of the size of the mask.
- `fill_mask_holes`: Whether to fully fill any holes (small or large) in the mask, that is, mark fully enclosed areas as part of the mask.
- `blur_mask_pixels`: Grows the mask and blurs it by the specified amount of pixels.
- `invert_mask`: Whether to fully invert the mask, that is, only keep what was marked, instead of removing what was marked.
- `blend_pixels`: Grows the stitch mask and blurs it by the specified amount of pixels, so that the stitch is slowly blended and there are no seams.
- `rescale_algorithm`: Rescale algorithm to use. bislerp is for super high quality but very slow, recommended for stich. bicubic is high quality and faster, recommended for crop.
- `mode`: Free size, Forced size, or Ranged size.
    - Ranged size upscales the area as much as possible to make it fit the larger size between `min_width`, `max_width`, `min_height`, and `max_height`, with a `padding` to align to standard sizes, then rescales before stitching back.
    - Forced size uses `force_width` and `force_height` and upscales the content to take that size before sampling, then downscales before stitching back. Use forced size e.g. for SDXL.
    - Free size uses `rescale_factor` to optionally rescale the content before sampling and eventually scale back before stitching, and `padding` to align to standard sizes.

## Example
This example inpaints by sampling on a small section of the larger image, upscaling to fit 512x512-768x768, then stitching and blending back in the original image.

Download the following example workflow from [here](inpaint-cropandstitch_example_workflow.json) or drag and drop the screenshot into ComfyUI.

![Workflow](inpaint-cropandstitch_example_workflow.png)

# Installation Instructions

Install via ComfyUI-Manager or go to the custom_nodes/ directory and run ```$ git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git```

## Best Practices
Use an inpainting model e.g. lazymixRealAmateur_v40Inpainting.

Use "InpaintModelConditioning" instead of "VAE Encode (for Inpainting)" to be able to set denoise values lower than 1.

If you want to inpaint fast with SD 1.5, use ranged size with min width and height 512 and max width and height 768 with padding 32. Set high rescale_factor (e.g. 10), it will be adapted to the right resolution.

If you want to inpaint with SDXL, use forced size = 1024.

# Changelog
## 2024-06-07
- Added the "Extend Image for Outpainting" node that allows leveraging the power of Inpaint Crop and Stitch (rescaling, blur, blend, restitching) for Outpainting.
## 2024-06-07
- Added a blending radius for seamless inpainting.
- Added a blur mask setting that grows and blurs the mask, providing better support
## 2024-06-01
- Force_size is now specified as separate force_width and force_height, to match any desired sampling resolution.
- New mode: ranged size, similar to free size but also takes min_width, min_height, max_width, and max_height, in order to avoid over scaling or under scaling beyond desirable limits.
## 2024-05-15
- Depending on the selected mode ("free size" or "forced size") some fields are hidden.
## 2024-05-14
- Batch support.
- Enabled selecting rescaling algorithm and made bicubic the default for crop, which significantly speeds up the process.
## 2024-05-13
- Switched from adjust_to_preferred_sizes to modes: free size and forced size. Forced scales the section rather than growing the context area to fit preferred_sizes, to be used to e.g. force 1024x1024 for inpainting.
- Enable internal_upscale_factor to be lower than 1 (that is, downscale), which can be used to avoid the double head issue in some models.
- Added padding on the croppedp image to avoid artifacts when the cropped image is not multiple of (default) 32
## 2024-05-12
- Add internal_upscale_factor to upscale the image before sampling and then downsizes to stitch it back.
## 2024-05-11
- Initial commit.

# Acknowledgements

This repository uses some code from comfy_extras (https://github.com/comfyanonymous/ComfyUI), KJNodes (https://github.com/kijai/ComfyUI-KJNodes), and Efficiency Nodes (https://github.com/LucianoCirino/efficiency-nodes-comfyui), all of them licensed under GNU GENERAL PUBLIC LICENSE Version 3. 

# License
GNU GENERAL PUBLIC LICENSE Version 3, see [LICENSE](LICENSE)
