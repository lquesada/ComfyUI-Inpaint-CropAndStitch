import comfy.utils
import nodes
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, grey_dilation, binary_fill_holes, binary_closing

class InpaintCrop:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch

    This node crop before sampling and stitch after sampling for fast, efficient inpainting without altering unmasked areas.
    Context area can be specified via expand pixels and expand factor or via a separate (optional) mask.
    Mask can be grown, holes in it filled, blurred, adjusted to preferred sizes or made square.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "context_expand_pixels": ("INT", {"default": 10, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "context_expand_factor": ("FLOAT", {"default": 1.01, "min": 1.0, "max": 100.0, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "grow_mask_pixels": ("INT", {"default": 12.0, "min": 0.0, "max": 1000, "step": 1}),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "blur_radius_pixels": ("FLOAT", {"default": 3.0, "min": 0.0, "max": nodes.MAX_RESOLUTION, "step": 0.1}),
                "adjust_to_preferred_sizes": ("BOOLEAN", {"default": False}),
                "preferred_sizes": ("STRING", {"default": "1024"}),
                "prefer_square_size": ("BOOLEAN", {"default": False}),
                "internal_upscale_factor": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 100.0, "step": 0.01}),
           },
           "optional": {
                "optional_context_mask": ("MASK",),
           }
        }

    CATEGORY = "inpaint"

    RETURN_TYPES = ("STITCH", "IMAGE", "MASK")
    RETURN_NAMES = ("stitch", "cropped_image", "cropped_mask")

    FUNCTION = "inpaint_crop"

    def adjust_to_preferred_size(self, min_val, max_val, max_dimension, preferred_size):
        # Calculate the new min and max to center the preferred size around the current center
        center = (min_val + max_val) // 2
        new_min = center - preferred_size // 2
        new_max = new_min + preferred_size - 1
    
        # Adjust to ensure the coordinates do not exceed the image boundaries
        if new_min < 0:
            new_min = 0
            new_max = preferred_size - 1
        if new_max >= max_dimension:
            new_max = max_dimension - 1
            new_min = new_max - preferred_size + 1
    
        return new_min, new_max

    # Parts of this function are from KJNodes: https://github.com/kijai/ComfyUI-KJNodes
    def inpaint_crop(self, image, mask, context_expand_pixels, context_expand_factor, invert_mask, grow_mask_pixels, fill_holes, blur_radius_pixels, adjust_to_preferred_sizes, preferred_sizes, prefer_square_size, internal_upscale_factor, optional_context_mask = None):
        original_image = image
        original_mask = mask
        original_width = image.shape[2]
        original_height = image.shape[1]

        # Invert mask if requested
        if invert_mask:
            mask = 1.0 - mask

        # Grow mask if requested
        if grow_mask_pixels > 0:
            growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in growmask:
                mask_np = m.numpy()
                kernel_size = grow_mask_pixels * 2 + 1
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                dilated_mask = grey_dilation(mask_np, footprint=kernel)
                output = dilated_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)

        # Fill holes if requested
        if fill_holes:
            holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in holemask:
                mask_np = m.numpy()
                binary_mask = mask_np > 0
                struct = np.ones((5, 5))
                closed_mask = binary_closing(binary_mask, structure=struct, border_value=1)
                filled_mask = binary_fill_holes(closed_mask)
                output = filled_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)

        # Blur mask if requested
        if blur_radius_pixels > 0:
            mask_np = mask.numpy()
            sigma = blur_radius_pixels / 2
            filtered_mask = gaussian_filter(mask_np, sigma=sigma)
            mask = torch.from_numpy(filtered_mask)
            mask = torch.clamp(mask, 0.0, 1.0)

        # Upscale image and masks if requested, they will be downsized at stitch phase
        effective_upscale_factor_x = 1.0
        effective_upscale_factor_y = 1.0
        if internal_upscale_factor < 0.999 or internal_upscale_factor > 1.001:
            samples = image            
            samples = samples.movedim(-1, 1)
            width = round(samples.shape[3] * internal_upscale_factor)
            height = round(samples.shape[2] * internal_upscale_factor)
            samples = comfy.utils.bislerp(samples, width, height)
            effective_upscale_factor_x = float(width)/float(original_width)
            effective_upscale_factor_y = float(height)/float(original_height)
            samples = samples.movedim(1, -1)
            image = samples

            samples = mask
            samples = samples.unsqueeze(1)
            samples = comfy.utils.bislerp(samples, width, height)
            samples = samples.squeeze(1)
            mask = samples

            if optional_context_mask is not None:
                samples = optional_context_mask
                samples = samples.unsqueeze(1)
                samples = comfy.utils.bislerp(samples, width, height)
                samples = samples.squeeze(1)
                optional_context_mask = samples

        # Set context mask if undefined. If present, expand with mask
        if optional_context_mask is None:
            context_mask = mask
        else:
            context_mask = optional_context_mask + mask 
            context_mask = torch.clamp(context_mask, 0.0, 1.0)

        non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)
        if not non_zero_indices[0].size(0):
            # If there are no non-zero indices, return the original image and original mask
            stitch = {'x': 0, 'y': 0, 'original_image': original_image, 'cropped_mask': mask, 'rescale_x': effective_upscale_factor_x, 'rescale_y': effective_upscale_factor_y}
            return (stitch, original_image, original_mask)

        # Compute context area from context mask
        y_min = torch.min(non_zero_indices[0]).item()
        y_max = torch.max(non_zero_indices[0]).item()
        x_min = torch.min(non_zero_indices[1]).item()
        x_max = torch.max(non_zero_indices[1]).item()
        height = context_mask.shape[1]
        width = context_mask.shape[2]
        
        # Grow context area if requested
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1
        y_grow = round(max(y_size*(context_expand_factor-1), context_expand_pixels))
        x_grow = round(max(x_size*(context_expand_factor-1), context_expand_pixels))
        y_min = max(y_min - y_grow // 2, 0)
        y_max = min(y_max + y_grow // 2, height - 1)
        x_min = max(x_min - x_grow // 2, 0)
        x_max = min(x_max + x_grow // 2, width - 1)

        # Adjust to the smallest preferred size larger than the current size if possible
        if adjust_to_preferred_sizes:
            preferred_sizes_parsed = [int(size) for size in preferred_sizes.split(',')]
            preferred_sizes_parsed.sort()
            preferred_x_size = next((size for size in preferred_sizes_parsed if size > x_size), None)
            preferred_y_size = next((size for size in preferred_sizes_parsed if size > y_size), None)
            if preferred_x_size is None:
                preferred_x_size = x_size
            if preferred_y_size is None:
                preferred_y_size = y_size
            if prefer_square_size:
                preferred_x_size = preferred_y_size = max(preferred_x_size, preferred_y_size)
            if preferred_x_size > width:
                preferred_x_size = width
            if preferred_y_size > height:
                preferred_y_size = height
            if preferred_x_size is not None:
                x_min, x_max = self.adjust_to_preferred_size(x_min, x_max, width, preferred_x_size)
            if preferred_y_size is not None:
                y_min, y_max = self.adjust_to_preferred_size(y_min, y_max, height, preferred_y_size)
        elif prefer_square_size:
            preferred_x_size = preferred_y_size = max(x_size, y_size)
            if preferred_x_size > width:
                preferred_x_size = width
            if preferred_y_size > height:
                preferred_y_size = height
            x_min, x_max = self.adjust_to_preferred_size(x_min, x_max, width, preferred_x_size)
            y_min, y_max = self.adjust_to_preferred_size(y_min, y_max, height, preferred_y_size)

        # Crop the image and the mask, sized context area
        cropped_image = image[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[:, y_min:y_max+1, x_min:x_max+1]

        # Return stitch (to be consumed by the class below), image, and mask
        stitch = {'x': x_min, 'y': y_min, 'original_image': original_image, 'cropped_mask': cropped_mask, 'rescale_x': effective_upscale_factor_x, 'rescale_y': effective_upscale_factor_y}
        return (stitch, cropped_image, cropped_mask)

class InpaintStitch:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch

    This node stitches the inpainted image without altering unmasked areas.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitch": ("STITCH",),
                "inpainted_image": ("IMAGE",),
            }
        }

    CATEGORY = "inpaint"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "inpaint_stitch"

    # This function is from comfy_extras: https://github.com/comfyanonymous/ComfyUI
    def composite(self, destination, source, x, y, mask = None, multiplier = 8, resize_source = False):
        source = source.to(destination.device)
        if resize_source:
            source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

        source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

        x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
        y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

        left, top = (x // multiplier, y // multiplier)
        right, bottom = (left + source.shape[3], top + source.shape[2],)

        if mask is None:
            mask = torch.ones_like(source)
        else:
            mask = mask.to(destination.device, copy=True)
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
            mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

        # calculate the bounds of the source that will be overlapping the destination
        # this prevents the source trying to overwrite latent pixels that are out of bounds
        # of the destination
        visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

        mask = mask[:, :, :visible_height, :visible_width]
        inverse_mask = torch.ones_like(mask) - mask
            
        source_portion = mask * source[:, :, :visible_height, :visible_width]
        destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

        destination[:, :, top:bottom, left:right] = source_portion + destination_portion
        return destination

    def inpaint_stitch(self, stitch, inpainted_image):
        original_image = stitch['original_image']
        cropped_mask = stitch['cropped_mask']
        x = stitch['x']
        y = stitch['y']
        stitched_image = original_image.clone().movedim(-1, 1)

        inpaint_width = inpainted_image.shape[2]
        inpaint_height = inpainted_image.shape[1]

        # Downscale inpainted before stitching if we upscaled it before
        if stitch['rescale_x'] < 0.999 or stitch['rescale_x'] > 1.001 or stitch['rescale_y'] < 0.999 or stitch['rescale_y'] > 1.001:
            samples = inpainted_image.movedim(-1, 1)
            width = round(float(inpaint_width)/stitch['rescale_x'])
            height = round(float(inpaint_height)/stitch['rescale_y'])
            x = round(float(x)/stitch['rescale_x'])
            y = round(float(y)/stitch['rescale_y'])
            samples = comfy.utils.bislerp(samples, width, height)
            inpainted_image = samples.movedim(1, -1)
            
            samples = cropped_mask.movedim(-1, 1)
            samples = samples.unsqueeze(0)
            samples = comfy.utils.bislerp(samples, width, height)
            samples = samples.squeeze(0)
            cropped_mask = samples.movedim(1, -1)

        output = self.composite(stitched_image, inpainted_image.movedim(-1, 1), x, y, cropped_mask, 1).movedim(1, -1)

        return (output,)
