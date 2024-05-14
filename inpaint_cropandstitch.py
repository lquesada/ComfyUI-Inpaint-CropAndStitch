import comfy.utils
import math
import nodes
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, grey_dilation, binary_fill_holes, binary_closing

def rescale(samples, width, height, algorithm):
    if algorithm == "nearest":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="nearest")
    elif algorithm == "bilinear":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="bilinear")
    elif algorithm == "bicubic":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="bicubic")
    elif algorithm == "bislerp":
        return comfy.utils.bislerp(samples, width, height)
    return samples

class InpaintCrop:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch

    This node crop before sampling and stitch after sampling for fast, efficient inpainting without altering unmasked areas.
    Context area can be specified via expand pixels and expand factor or via a separate (optional) mask.
    Works free size and also forced size.
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
                "fill_mask_holes": ("BOOLEAN", {"default": True}),
                "rescale_algorithm": (["nearest", "bilinear", "bicubic", "bislerp"], {"default": "bicubic"}),
                "mode": (["free size", "forced size"], {"default": "free size"}),
                "force_size": ([512, 768, 1024, 1344, 2048, 4096, 8192], {"default": 1024}),
                "rescale_factor": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 100.0, "step": 0.01}),
                "padding": ([8, 16, 32, 64, 128, 256, 512], {"default": 32}),
           },
           "optional": {
                "optional_context_mask": ("MASK",),
           }
        }

    CATEGORY = "inpaint"

    RETURN_TYPES = ("STITCH", "IMAGE", "MASK")
    RETURN_NAMES = ("stitch", "cropped_image", "cropped_mask")
    OUTPUT_IS_LIST = (True, True, True)

    FUNCTION = "inpaint_crop"

    def adjust_to_square(self, x_min, x_max, y_min, y_max, width, height, target_size=None):
        if target_size is None:
            x_size = x_max - x_min + 1
            y_size = y_max - y_min + 1
            target_size = max(x_size, y_size)

        # Calculate the midpoint of the current x and y ranges
        x_mid = (x_min + x_max) // 2
        y_mid = (y_min + y_max) // 2

        # Adjust x_min, x_max, y_min, y_max to make the range square centered around the midpoints
        x_min = max(x_mid - target_size // 2, 0)
        x_max = x_min + target_size - 1
        y_min = max(y_mid - target_size // 2, 0)
        y_max = y_min + target_size - 1

        # Ensure the ranges do not exceed the image boundaries
        if x_max >= width:
            x_max = width - 1
            x_min = x_max - target_size + 1
        if y_max >= height:
            y_max = height - 1
            y_min = y_max - target_size + 1

        # Additional checks to make sure all coordinates are within bounds
        if x_min < 0:
            x_min = 0
            x_max = target_size - 1
        if y_min < 0:
            y_min = 0
            y_max = target_size - 1

        return x_min, x_max, y_min, y_max

    def apply_padding(self, min_val, max_val, max_boundary, padding):
        # Calculate the midpoint and the original range size
        original_range_size = max_val - min_val + 1
        midpoint = (min_val + max_val) // 2

        # Determine the smallest multiple of padding that is >= original_range_size
        if original_range_size % padding == 0:
            new_range_size = original_range_size
        else:
            new_range_size = (original_range_size // padding + 1) * padding

        # Calculate the new min and max values centered on the midpoint
        new_min_val = max(midpoint - new_range_size // 2, 0)
        new_max_val = new_min_val + new_range_size - 1

        # Ensure the new max doesn't exceed the boundary
        if new_max_val >= max_boundary:
            new_max_val = max_boundary - 1
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        # Ensure the range still ends on a multiple of padding
        # Adjust if the calculated range isn't feasible within the given constraints
        if (new_max_val - new_min_val + 1) != new_range_size:
            new_min_val = max(new_max_val - new_range_size + 1, 0)

        return new_min_val, new_max_val

    def inpaint_crop(self, image, mask, context_expand_pixels, context_expand_factor, invert_mask, fill_mask_holes, mode, rescale_algorithm, force_size, rescale_factor, padding, optional_context_mask=None):
        assert image.shape[0] == mask.shape[0], "Batch size of images and masks must be the same"
        if optional_context_mask is not None:
            assert optional_context_mask.shape[0] == image.shape[0], "Batch size of optional_context_masks must be the same as images or None"

        results_stitch = []
        results_image = []
        results_mask = []

        batch_size = image.shape[0]
        for b in range(batch_size):
            one_image = image[b].unsqueeze(0)
            one_mask = mask[b].unsqueeze(0)
            one_optional_context_mask = None
            if optional_context_mask is not None:
                one_optional_context_mask = optional_context_mask[b].unsqueeze(0)

            stitch, cropped_image, cropped_mask = self.inpaint_crop_single_image(
                one_image, one_mask, context_expand_pixels, context_expand_factor, invert_mask,
                fill_mask_holes, mode, rescale_algorithm, force_size, rescale_factor,
                padding, one_optional_context_mask
            )

            results_stitch.append(stitch)
            results_image.append(cropped_image)
            results_mask.append(cropped_mask)

        return results_stitch, results_image, results_mask
       
    # Parts of this function are from KJNodes: https://github.com/kijai/ComfyUI-KJNodes
    def inpaint_crop_single_image(self, image, mask, context_expand_pixels, context_expand_factor, invert_mask, fill_mask_holes, mode, rescale_algorithm, force_size, rescale_factor, padding, optional_context_mask=None):
        original_image = image
        original_mask = mask
        original_width = image.shape[2]
        original_height = image.shape[1]

        #Validate or initialize mask
        if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                mask = torch.zeros_like(image[:, :, :, 0])
            else:
                assert False, "mask size must match image size"

        # Invert mask if requested
        if invert_mask:
            mask = 1.0 - mask

        # Fill holes if requested
        if fill_mask_holes:
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

        # Validate or initialize context mask
        if optional_context_mask is None:
            context_mask = mask
        elif optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(optional_context_mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                context_mask = mask
            else:
                assert False, "context_mask size must match image size"
        else:
            context_mask = optional_context_mask + mask 
            context_mask = torch.clamp(context_mask, 0.0, 1.0)

        # If there are no non-zero indices in the context_mask, return the original image and original mask
        non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)
        if not non_zero_indices[0].size(0):
            stitch = {'x': 0, 'y': 0, 'original_image': original_image, 'cropped_mask': mask, 'rescale_x': 1.0, 'rescale_y': 1.0}
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

        effective_upscale_factor_x = 1.0
        effective_upscale_factor_y = 1.0
        # Adjust to preferred size
        if mode == 'forced size':
            # Turn into square
            x_min, x_max, y_min, y_max = self.adjust_to_square(x_min, x_max, y_min, y_max, width, height)
            current_size = x_max - x_min + 1  # Assuming x_max - x_min == y_max - y_min due to square adjustment
            if current_size != force_size:
                # Upscale to fit in the force_size square, will be downsized at stitch phase
                upscale_factor = force_size / current_size

                samples = image            
                samples = samples.movedim(-1, 1)

                width = math.floor(samples.shape[3] * upscale_factor)
                height = math.floor(samples.shape[2] * upscale_factor)
                samples = rescale(samples, width, height, rescale_algorithm)
                effective_upscale_factor_x = float(width)/float(original_width)
                effective_upscale_factor_y = float(height)/float(original_height)
                samples = samples.movedim(1, -1)
                image = samples

                samples = mask
                samples = samples.unsqueeze(1)
                samples = rescale(samples, width, height, rescale_algorithm)
                samples = samples.squeeze(1)
                mask = samples

                x_min = math.floor(x_min * effective_upscale_factor_x)
                x_max = math.floor(x_max * effective_upscale_factor_x)
                y_min = math.floor(y_min * effective_upscale_factor_y)
                y_max = math.floor(y_max * effective_upscale_factor_y)

                # Readjust to force size because the upscale math may not round well
                x_min, x_max, y_min, y_max = self.adjust_to_square(x_min, x_max, y_min, y_max, width, height, target_size=force_size)

        elif mode == 'free size':
            # Upscale image and masks if requested, they will be downsized at stitch phase
            if rescale_factor < 0.999 or rescale_factor > 1.001:
                samples = image            
                samples = samples.movedim(-1, 1)

                width = math.floor(samples.shape[3] * rescale_factor)
                height = math.floor(samples.shape[2] * rescale_factor)
                samples = rescale(samples, width, height, rescale_algorithm)
                effective_upscale_factor_x = float(width)/float(original_width)
                effective_upscale_factor_y = float(height)/float(original_height)
                samples = samples.movedim(1, -1)
                image = samples

                samples = mask
                samples = samples.unsqueeze(1)
                samples = rescale(samples, width, height, rescale_algorithm)
                samples = samples.squeeze(1)
                mask = samples

                x_min = math.floor(x_min * effective_upscale_factor_x)
                x_max = math.floor(x_max * effective_upscale_factor_x)
                y_min = math.floor(y_min * effective_upscale_factor_y)
                y_max = math.floor(y_max * effective_upscale_factor_y)

                # Ensure that context area doesn't go outside of the image
                x_min = max(x_min, 0)
                x_max = min(x_max, width - 1)
                y_min = max(y_min, 0)
                y_max = min(y_max, height - 1)

            # Pad area (if possible, i.e. if pad is smaller than width/height) to avoid the sampler returning smaller results
            if padding > 1:
                x_min, x_max = self.apply_padding(x_min, x_max, width, padding)
                y_min, y_max = self.apply_padding(y_min, y_max, height, padding)


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
                "rescale_algorithm": (["nearest", "bilinear", "bicubic", "bislerp"], {"default": "bislerp"}),
            }
        }

    CATEGORY = "inpaint"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "inpaint_stitch"

    # This function is from comfy_extras: https://github.com/comfyanonymous/ComfyUI
    def composite(self, destination, source, x, y, mask=None, multiplier=8, resize_source=False):
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

    def inpaint_stitch(self, stitch, inpainted_image, rescale_algorithm):
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
            samples = rescale(samples, width, height, rescale_algorithm)
            inpainted_image = samples.movedim(1, -1)
            
            samples = cropped_mask.movedim(-1, 1)
            samples = samples.unsqueeze(0)
            samples = rescale(samples, width, height, rescale_algorithm)
            samples = samples.squeeze(0)
            cropped_mask = samples.movedim(1, -1)

        output = self.composite(stitched_image, inpainted_image.movedim(-1, 1), x, y, cropped_mask, 1).movedim(1, -1)

        return (output,)
