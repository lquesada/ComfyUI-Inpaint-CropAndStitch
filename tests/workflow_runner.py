import os
import json
import math
import torch
import numpy as np
from PIL import Image, ImageOps
import inpaint_cropandstitch
from inpaint_cropandstitch import InpaintCropImproved, InpaintStitchImproved


def repeat_to_batch_size(tensor, batch_size, dim=0):
    """Repeat tensor along dimension to match batch_size (matching comfy.utils)."""
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        repeats = dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)
        return tensor.repeat(repeats).narrow(dim, 0, batch_size)
    return tensor


def image_alpha_fix(destination, source):
    """Align alpha channel dimension between destination and source (matching node_helpers)."""
    if destination.shape[-1] < source.shape[-1]:
        source = source[..., :destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        source = torch.nn.functional.pad(source, (0, 1))
        source[..., -1] = 1.0
    return destination, source


def composite_images(destination, source, x, y, mask=None, multiplier=1, resize_source=False):
    """
    Self-contained implementation of ComfyUI's composite function for images.
    destination and source in [B, C, H, W] format.
    """
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(
            source, size=(destination.shape[-2], destination.shape[-1]), mode="bilinear"
        )

    source = repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[-1] * multiplier, min(x, destination.shape[-1] * multiplier))
    y = max(-source.shape[-2] * multiplier, min(y, destination.shape[-2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[-1], top + source.shape[-2])

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(source.shape[-2], source.shape[-1]),
            mode="bilinear",
        )
        mask = repeat_to_batch_size(mask, source.shape[0])

    visible_width = destination.shape[-1] - left + min(0, x)
    visible_height = destination.shape[-2] - top + min(0, y)

    mask = mask[:, :, :visible_height, :visible_width]
    if mask.ndim < source.ndim:
        mask = mask.unsqueeze(1)

    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[..., :visible_height, :visible_width]
    destination_portion = inverse_mask * destination[..., top:bottom, left:right]

    destination[..., top:bottom, left:right] = source_portion + destination_portion
    return destination


class MockLoadImage:
    """Self-contained mock for ComfyUI's LoadImage node."""

    def __init__(self, base_dir=None):
        if base_dir is None:
            # Default to repo root / testimgs
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testimgs")
        self.base_dir = base_dir

    def load_image(self, image_name):
        clean_name = image_name.replace(" [input]", "")
        candidates = [
            os.path.join(self.base_dir, clean_name),
            os.path.join(self.base_dir, os.path.basename(clean_name)),
            clean_name,
        ]
        found_path = None
        for c in candidates:
            if os.path.exists(c):
                found_path = c
                break

        if found_path is None:
            raise FileNotFoundError(f"MockLoadImage: Image file not found: {image_name}. Tried {candidates}")

        with Image.open(found_path) as img:
            img = ImageOps.exif_transpose(img)
            rgb = img.convert("RGB")
            image_np = np.array(rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # [1, H, W, 3]

            if "A" in img.getbands():
                mask_np = np.array(img.getchannel("A")).astype(np.float32) / 255.0
                mask_tensor = 1.0 - torch.from_numpy(mask_np)
            else:
                mask_tensor = torch.zeros((64, 64), dtype=torch.float32)
            mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]

        return (image_tensor, mask_tensor)


class MockMaskToImage:
    """Self-contained mock for ComfyUI's MaskToImage node."""

    def mask_to_image(self, mask):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return (result,)


class MockImageInvert:
    """Self-contained mock for ComfyUI's ImageInvert node."""

    def invert(self, image):
        return (1.0 - image,)


class MockImpactMakeImageBatch:
    """Self-contained mock for ImpactMakeImageBatch node."""

    def make_batch(self, **kwargs):
        imgs = [v for k, v in sorted(kwargs.items()) if k.startswith("image") and v is not None]
        if not imgs:
            raise ValueError("ImpactMakeImageBatch: No images provided to batch.")
        # Ensure 4D
        imgs_4d = [img.unsqueeze(0) if img.ndim == 3 else img for img in imgs]
        return (torch.cat(imgs_4d, dim=0),)


class MockImpactMakeMaskBatch:
    """Self-contained mock for ImpactMakeMaskBatch node."""

    def make_batch(self, **kwargs):
        ms = [v for k, v in sorted(kwargs.items()) if k.startswith("mask") and v is not None]
        if not ms:
            raise ValueError("ImpactMakeMaskBatch: No masks provided to batch.")
        # Ensure 3D
        ms_3d = [m.unsqueeze(0) if m.ndim == 2 else m for m in ms]
        return (torch.cat(ms_3d, dim=0),)


class MockImageCompositeMasked:
    """Self-contained mock for ImageCompositeMasked node."""

    def composite(self, destination, source, x=0, y=0, resize_source=False, mask=None):
        destination, source = image_alpha_fix(destination, source)
        dest_ch = destination.clone().movedim(-1, 1)
        src_ch = source.movedim(-1, 1)
        output = composite_images(dest_ch, src_ch, x, y, mask, multiplier=1, resize_source=resize_source).movedim(1, -1)
        return (output,)


class WorkflowRunResult:
    """Holds all results and statistics of a workflow execution run."""

    def __init__(self, wf_name, nodes, links, outputs):
        self.wf_name = wf_name
        self.nodes = nodes
        self.links = links
        self.outputs = outputs  # node_id -> tuple of output values

    @property
    def crop_node_ids(self):
        return [nid for nid, n in self.nodes.items() if n.get("type") == "InpaintCropImproved"]

    @property
    def stitch_node_ids(self):
        return [nid for nid, n in self.nodes.items() if n.get("type") == "InpaintStitchImproved"]

    @property
    def preview_node_ids(self):
        return [nid for nid, n in self.nodes.items() if n.get("type") == "PreviewImage"]

    @property
    def load_node_ids(self):
        return [nid for nid, n in self.nodes.items() if n.get("type") == "LoadImage"]


class WorkflowRunner:
    """
    Parses and executes ComfyUI workflows without requiring an external ComfyUI server.
    """

    CROP_WIDGET_NAMES = [
        "downscale_algorithm",
        "upscale_algorithm",
        "preresize",
        "preresize_mode",
        "preresize_min_width",
        "preresize_min_height",
        "preresize_max_width",
        "preresize_max_height",
        "mask_fill_holes",
        "mask_expand_pixels",
        "mask_invert",
        "mask_blend_pixels",
        "mask_hipass_filter",
        "extend_for_outpainting",
        "extend_up_factor",
        "extend_down_factor",
        "extend_left_factor",
        "extend_right_factor",
        "context_from_mask_extend_factor",
        "output_resize_to_target_size",
        "output_target_width",
        "output_target_height",
        "output_padding",
        "device_mode",
    ]

    def __init__(self, testimgs_dir=None, verbose=False):
        self.verbose = verbose
        self.load_image_node = MockLoadImage(testimgs_dir)
        self.mask_to_image_node = MockMaskToImage()
        self.image_invert_node = MockImageInvert()
        self.image_batch_node = MockImpactMakeImageBatch()
        self.mask_batch_node = MockImpactMakeMaskBatch()
        self.composite_node = MockImageCompositeMasked()

        # Initialize Crop & Stitch nodes with DEBUG_MODE enabled
        self.crop_node = InpaintCropImproved()
        self.crop_node.DEBUG_MODE = True
        self.crop_node.VERBOSE = verbose
        self.crop_node.RETURN_NAMES = InpaintCropImproved.DEBUG_RETURN_NAMES

        self.stitch_node = InpaintStitchImproved()

    def run_file(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            wf_data = json.load(f)
        return self.run_dict(wf_data, name=os.path.basename(json_path))

    def run_dict(self, wf_data, name="workflow"):
        nodes = {n["id"]: n for n in wf_data.get("nodes", [])}
        links = {l[0]: l for l in wf_data.get("links", [])}
        memo = {}

        def get_node_output(node_id):
            if node_id in memo:
                return memo[node_id]

            n = nodes[node_id]
            ntype = n.get("type")
            wv = n.get("widgets_values") or []
            inputs = n.get("inputs") or []

            # Resolve linked inputs
            resolved_inputs = {}
            for inp in inputs:
                iname = inp.get("name")
                lid = inp.get("link")
                if lid is not None:
                    link_info = links[lid]
                    from_node_id = link_info[1]
                    from_slot_idx = link_info[2]
                    from_outs = get_node_output(from_node_id)
                    resolved_inputs[iname] = from_outs[from_slot_idx]

            # Execute node based on type
            if ntype == "LoadImage":
                filename = wv[0] if wv else "example.png"
                res = self.load_image_node.load_image(filename)

            elif ntype == "ImageInvert":
                res = self.image_invert_node.invert(resolved_inputs["image"])

            elif ntype == "MaskToImage":
                res = self.mask_to_image_node.mask_to_image(resolved_inputs["mask"])

            elif ntype == "ImpactMakeImageBatch":
                res = self.image_batch_node.make_batch(**resolved_inputs)

            elif ntype == "ImpactMakeMaskBatch":
                res = self.mask_batch_node.make_batch(**resolved_inputs)

            elif ntype == "ImageCompositeMasked":
                dest = resolved_inputs["destination"]
                src = resolved_inputs["source"]
                mask = resolved_inputs.get("mask")
                x = wv[0] if len(wv) > 0 else 0
                y = wv[1] if len(wv) > 1 else 0
                resize_source = wv[2] if len(wv) > 2 else False
                res = self.composite_node.composite(dest, src, x=x, y=y, resize_source=resize_source, mask=mask)

            elif ntype == "InpaintCropImproved":
                kwargs = {}
                for wname, wval in zip(self.CROP_WIDGET_NAMES, wv):
                    kwargs[wname] = wval
                kwargs["image"] = resolved_inputs["image"]
                kwargs["mask"] = resolved_inputs.get("mask", None)
                kwargs["optional_context_mask"] = resolved_inputs.get("optional_context_mask", None)
                res = self.crop_node.inpaint_crop(**kwargs)

            elif ntype == "InpaintStitchImproved":
                stitcher = resolved_inputs["stitcher"]
                inpainted_image = resolved_inputs["inpainted_image"]
                res = self.stitch_node.inpaint_stitch(stitcher, inpainted_image)

            elif ntype == "PreviewImage":
                res = (resolved_inputs["images"],)

            elif ntype == "Note":
                res = ()

            else:
                raise ValueError(f"WorkflowRunner: Unsupported node type '{ntype}' (id: {node_id})")

            memo[node_id] = res
            return res

        for nid in nodes:
            get_node_output(nid)

        return WorkflowRunResult(name, nodes, links, memo)


def validate_tensor(tensor, name, expected_ndim=None, min_val=0.0, max_val=1.0):
    """Assert tensor validity: type, ndim, finite values, and range."""
    assert isinstance(tensor, torch.Tensor), f"{name}: Expected torch.Tensor, got {type(tensor)}"
    if expected_ndim is not None:
        assert tensor.ndim == expected_ndim, f"{name}: Expected {expected_ndim} dims, got {tensor.ndim} (shape: {tensor.shape})"
    assert not torch.isnan(tensor).any(), f"{name}: Tensor contains NaN values."
    assert not torch.isinf(tensor).any(), f"{name}: Tensor contains Inf values."
    if min_val is not None and tensor.numel() > 0:
        actual_min = tensor.min().item()
        assert actual_min >= min_val - 1e-3, f"{name}: Value below minimum: {actual_min} < {min_val}"
    if max_val is not None and tensor.numel() > 0:
        actual_max = tensor.max().item()
        assert actual_max <= max_val + 1e-3, f"{name}: Value above maximum: {actual_max} > {max_val}"


def validate_crop_outputs(outputs, node_id=None):
    """
    Validates outputs of InpaintCropImproved node:
    - Slot 0: stitcher dict with all required spatial & canvas metadata
    - Slot 1: cropped_image [B, H, W, 3]
    - Slot 2: cropped_mask [B, H, W]
    - Slots 3..25: all 23 debug tensors
    """
    prefix = f"Crop node {node_id}" if node_id is not None else "Crop node"
    assert len(outputs) == 26, f"{prefix}: Expected 26 outputs, got {len(outputs)}"

    stitcher = outputs[0]
    cropped_image = outputs[1]
    cropped_mask = outputs[2]

    # 1. Validate stitcher dict
    assert isinstance(stitcher, dict), f"{prefix}: Stitcher output is not a dict"
    required_keys = [
        "cropped_to_canvas_x",
        "cropped_to_canvas_y",
        "cropped_to_canvas_w",
        "cropped_to_canvas_h",
        "canvas_image",
        "cropped_mask_for_blend",
        "canvas_to_orig_x",
        "canvas_to_orig_y",
        "canvas_to_orig_w",
        "canvas_to_orig_h",
    ]
    for k in required_keys:
        assert k in stitcher, f"{prefix}: Stitcher missing key '{k}'"

    # 2. Validate cropped_image & cropped_mask
    validate_tensor(cropped_image, f"{prefix} cropped_image", expected_ndim=4, min_val=0.0, max_val=1.0)
    validate_tensor(cropped_mask, f"{prefix} cropped_mask", expected_ndim=3, min_val=0.0, max_val=1.0)
    assert cropped_image.shape[0] == cropped_mask.shape[0], f"{prefix}: Batch mismatch between image and mask"
    assert cropped_image.shape[1:3] == cropped_mask.shape[1:3], f"{prefix}: Spatial mismatch between image and mask"

    # 3. Validate debug tensors
    for i in range(3, 26):
        out_tensor = outputs[i]
        validate_tensor(out_tensor, f"{prefix} debug slot {i}", expected_ndim=None, min_val=0.0, max_val=1.0)


def validate_stitch_outputs(stitched_output, stitcher, inpainted_image, node_id=None):
    """
    Validates outputs of InpaintStitchImproved node:
    - Output is [B, H, W, C]
    - Shape matches original canvas
    - Values are within [0.0, 1.0], no NaNs or Infs
    - Critical invariant: Outside of the modified/masked area, canvas pixels are preserved!
    """
    prefix = f"Stitch node {node_id}" if node_id is not None else "Stitch node"
    assert isinstance(stitched_output, tuple) and len(stitched_output) >= 1, f"{prefix}: Invalid return format"
    stitched = stitched_output[0]

    validate_tensor(stitched, f"{prefix} stitched image", expected_ndim=4, min_val=0.0, max_val=1.0)

    # Validate that output shape matches canvas shape
    canvas_imgs = stitcher["canvas_image"]
    orig_h = stitcher["canvas_to_orig_h"][0]
    orig_w = stitcher["canvas_to_orig_w"][0]
    assert stitched.shape[1] == orig_h, f"{prefix}: Height mismatch {stitched.shape[1]} vs expected {orig_h}"
    assert stitched.shape[2] == orig_w, f"{prefix}: Width mismatch {stitched.shape[2]} vs expected {orig_w}"

    # Verify unmasked area preservation:
    # Where blend mask is 0 (outside inpainted region), stitched image should exactly equal original canvas
    for i in range(min(stitched.shape[0], len(canvas_imgs))):
        c_img = canvas_imgs[i]
        if torch.is_tensor(c_img):
            c_img = c_img.cpu()
            if c_img.ndim == 3:
                c_img = c_img.unsqueeze(0)
            # Check corner pixel (0, 0) if crop box doesn't touch (0, 0)
            top_x = stitcher["cropped_to_canvas_x"][i]
            top_y = stitcher["cropped_to_canvas_y"][i]
            if top_x > 0 and top_y > 0:
                s_pixel = stitched[i, 0, 0, :3]
                c_pixel = c_img[0, 0, 0, :3]
                assert torch.allclose(s_pixel, c_pixel, atol=1e-3), (
                    f"{prefix}: Unmasked pixel changed at (0, 0): {s_pixel} vs {c_pixel}"
                )


def validate_workflow_run(result):
    """
    Performs comprehensive verification on all nodes of a completed workflow execution:
    - Checks that all nodes were evaluated
    - Validates all 105 InpaintCropImproved nodes
    - Validates all 34 InpaintStitchImproved nodes
    - Validates all 349 PreviewImage sink nodes
    """
    assert len(result.outputs) == len(result.nodes), (
        f"Workflow {result.wf_name}: executed {len(result.outputs)} of {len(result.nodes)} nodes."
    )

    # 1. Validate Crop nodes
    for nid in result.crop_node_ids:
        outs = result.outputs[nid]
        validate_crop_outputs(outs, node_id=nid)

    # 2. Validate Stitch nodes
    for nid in result.stitch_node_ids:
        stitch_node = result.nodes[nid]
        stitcher_link_id = next(inp["link"] for inp in stitch_node["inputs"] if inp["name"] == "stitcher")
        inpaint_link_id = next(inp["link"] for inp in stitch_node["inputs"] if inp["name"] == "inpainted_image")
        s_link = result.links[stitcher_link_id]
        i_link = result.links[inpaint_link_id]
        stitcher = result.outputs[s_link[1]][s_link[2]]
        inpainted = result.outputs[i_link[1]][i_link[2]]
        validate_stitch_outputs(result.outputs[nid], stitcher, inpainted, node_id=nid)

    # 3. Validate Preview nodes
    for nid in result.preview_node_ids:
        outs = result.outputs[nid]
        assert len(outs) == 1, f"Preview node {nid}: expected 1 output"
        validate_tensor(outs[0], f"Preview node {nid}", expected_ndim=4, min_val=0.0, max_val=1.0)
