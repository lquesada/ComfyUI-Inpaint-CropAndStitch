ComfyUI-Inpaint-CropAndStitch

Copyright (c) 2024, Luis Quesada Torres - https://github.com/lquesada | www.luisquesada.com

Check ComfyUI here: https://github.com/comfyanonymous/ComfyUI

# Overview

"✂️  Inpaint Crop" is a node that crops an image before sampling. The context area can be specified via the mask, expand pixels and expand factor or via a separate (optional) mask.

"✂️  Inpaint Stitch" is a node that stitches the inpainted image back into the original image without altering unmasked areas.

## Parameters
- `context_expand_pixels`: how much to grow the context area (i.e. the area for the sampling) around the original mask, in pixels. This provides more context for the sampling.
- `context_expand_factor`: how much to grow the context area (i.e. the area for the sampling) around the original mask, as a factor, e.g. 1.1 is grow 10% of the size of the mask.
- `invert_mask`: Whether to fully invert the mask, that is, only keep what was marked, instead of removing what was marked.
- `grow_mask_pixels`: How many pixels to grow the mask to provide a bit of a border, to be used in combination with `blur_radius_pixels`.
- `fill_holes`: Whether to fully fill any holes (small or large) in the mask, that is, mark fully enclosed areas as part of the mask.
- `blur_radius_pixels`: Whether to blur the mask, to be used in combination with `grow_mask_pixels`. Some models prefer blurred masks, some don't.
- `adjust_to_preferred_sizes`: This will try to have width and/or height of the context area match any of the sizes in `preferred_sizes` by growing the context area (not upscaling), e.g. 512, 1024. Some models prefer this.
- `preferred_sizes`: Comma-separated list of preferred sizes, e.g. "512,1024". Default is 1024 because it fits most of the use cases of this feature.
- `prefer_square_size`: This will try to have width=height (if possible). Some models prefer this.
- `internal_upscale_factor`: Upscale the image and mask between the crop and stitch phases. This means the sampling happens only around the context area but at a higher resolution (e.g. 2 for x2), then it is downsampled and merged with the original image. This in practice gets more details from models. If you want to obtain a higher resolution image, please upscale it before cropping/sampling/stitching. This can also be lower than 1 for the rare cases you need it, such as the original image too large and the model generating double heads.
- `padding`: Pad the cropped image and mask to a certain number of pixels in order to avoid artifacts. 8, 16, 32 are reasonable values.

## Simple example
This example inpaints by sampling on a small section of the larger image. It runs ~20x faster than sampling on the whole image.

Download the following example workflow from [here](inpaint-cropandstitch_example_workflow.json) or drag and drop the screenshot into ComfyUI.

![Workflow](inpaint-cropandstitch_example_workflow.png)

## Advanced example
This example inpaints by sampling on a small section of the larger image, but expands the context using a second (optional) context mask. It runs ~10x faster than sampling on the whole image but allows navigating the tradeoff between context and efficiency.

Download the following example workflow from [here](inpaint-cropandstitch_example_workflow_advanced.json) or drag and drop the screenshot into ComfyUI.

![Workflow](inpaint-cropandstitch_example_workflow_advanced.png)

# Installation Instructions

Install via ComfyUI-Manager or go to the custom_nodes/ directory and run ```$ git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git```

# Changelog
## Upcoming!
- Make adjust_to_preferred_sizes scale the section rather than grow the context area to fit preferred_sizes, to be used to e.g. force 1024x1024 for inpainting.
## 2024-05-13
- Enable internal_upscale_factor to be lower than 1 (that is, downscale), which can be used to avoid the double head issue in some models.
- Added padding on the croppedp image to avoid artifacts when the cropped image is not multiple of (default) 32
## 2024-05-12
- Add internal_upscale_factor to upscale the image before sampling and then downsizes to stitch it back.
## 2024-05-11
- Initial commit.

# Acknowledgements

This repository uses some code from comfy_extras (https://github.com/comfyanonymous/ComfyUI) and from KJNodes (https://github.com/kijai/ComfyUI-KJNodes), both licensed under GNU GENERAL PUBLIC LICENSE Version 3. 

# License
GNU GENERAL PUBLIC LICENSE Version 3, see [LICENSE](LICENSE)
