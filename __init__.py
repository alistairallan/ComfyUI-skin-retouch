"""
ComfyUI Skin Retouch Node
Runs entirely on PyTorch + onnxruntime — no TensorFlow, no modelscope runtime.
All model files are loaded directly from the node's own `models/` folder.

Controls exposed in ComfyUI:
  retouch_degree      – strength of the skin-smoothing U-Net (0.0–1.0)
  whitening_degree    – strength of skin whitening via ONNX mask (0.0–1.0)
  enable_whitening    – toggle whitening on/off
  enable_local_retouch– toggle the local blemish-inpainting pass
  face_score_thresh   – minimum RetinaFace detection confidence to process a face
"""

import os
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import onnxruntime

# ── Model architecture imports (pure PyTorch, no TF) ─────────────────────────
# These live inside modelscope's installed package but are plain nn.Module code.
from modelscope.models.cv.skin_retouching.unet_deploy import UNet
from modelscope.models.cv.skin_retouching.detection_model.detection_unet_in import DetectionUNet
from modelscope.models.cv.skin_retouching.inpainting_model.inpainting_unet import RetouchingNet
from modelscope.models.cv.skin_retouching.utils import (
    resize_on_long_side, get_crop_bbox, get_roi_without_padding,
    roi_to_tensor, preprocess_roi, smooth_border_mg,
    patch_partition_overlap, patch_aggregation_overlap,
    gen_diffuse_mask, whiten_img,
)
from modelscope.outputs import OutputKeys


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def _model_path(filename):
    return os.path.join(MODELS_DIR, filename)


def _load_onnx(path):
    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers.insert(0, "CUDAExecutionProvider")
    sess = onnxruntime.InferenceSession(path, providers=providers)
    input_names  = [n.name for n in sess.get_inputs()]
    output_names = [n.name for n in sess.get_outputs()]
    return sess, input_names, output_names


def _detect_faces(detector, rgb_image, score_thresh):
    """Run RetinaFace and return list of dicts with bbox/score/landmarks."""
    det_results = detector(rgb_image)
    results = []
    for i, score in enumerate(det_results["scores"]):
        if score < score_thresh:
            continue
        results.append({
            "bbox":      np.array(det_results["boxes"][i]).astype(np.int32).tolist(),
            "score":     score,
            "landmarks": np.array(det_results["keypoints"][i]).astype(np.int32).reshape(5, 2).tolist(),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# The ComfyUI node
# ─────────────────────────────────────────────────────────────────────────────

class SkinRetouchingNode:
    """
    Skin retouching via local U-Net + optional skin-whitening ONNX mask.
    All inference is PyTorch / onnxruntime — no TensorFlow dependency.
    """

    def __init__(self):
        self._loaded = False
        # models are lazy-loaded on first execution
        self.generator       = None
        self.detector        = None
        self.inpainting_net  = None
        self.detection_net   = None
        self.onnx_sess       = None
        self.onnx_in         = None
        self.onnx_out        = None
        self.diffuse_mask    = None
        self.device          = None

    # ------------------------------------------------------------------
    def _load_models(self):
        if self._loaded:
            return

        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available()
                              else "cpu")
        self.device = device

        # 1. Main retouching U-Net
        self.generator = UNet(3, 3).to(device)
        ckpt = torch.load(_model_path("pytorch_model.pt"), map_location="cpu")
        self.generator.load_state_dict(ckpt["generator"])
        self.generator.eval()

        # 2. Face detector (RetinaFace weights bundled with modelscope)
        #    We use modelscope's pipeline only for the detector — it's pure PyTorch.
        from modelscope.pipelines import pipeline as ms_pipeline
        from modelscope.utils.constant import Tasks
        det_pipeline = ms_pipeline(
            Tasks.face_detection,
            model="damo/cv_resnet50_face-detection_retinaface",
        )
        self.detector = det_pipeline

        # 3. Local blemish detection + inpainting nets
        joint_ckpt = torch.load(_model_path("joint_20210926.pth"), map_location="cpu")
        self.inpainting_net = RetouchingNet(in_channels=4, out_channels=3).to(device)
        self.detection_net  = DetectionUNet(n_channels=3, n_classes=1).to(device)
        self.inpainting_net.load_state_dict(joint_ckpt["inpainting_net"])
        self.detection_net.load_state_dict(joint_ckpt["detection_net"])
        self.inpainting_net.eval()
        self.detection_net.eval()

        # 4. Skin-mask ONNX model
        self.onnx_sess, self.onnx_in, self.onnx_out = _load_onnx(
            _model_path("model.onnx"))

        # 5. Border-smoothing diffuse mask
        dm = gen_diffuse_mask()
        self.diffuse_mask = torch.from_numpy(dm).to(device).float()
        self.diffuse_mask = self.diffuse_mask.permute(2, 0, 1)[None, ...]

        self._loaded = True

    # ------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":              ("IMAGE",),
                "retouch_degree":     ("FLOAT",  {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                                   "tooltip": "Skin-smoothing strength. 0 = no change, 1 = full effect."}),
                "whitening_degree":   ("FLOAT",  {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                                   "tooltip": "Skin-whitening strength applied via ONNX mask."}),
                "enable_whitening":   ("BOOLEAN",{"default": True,
                                                   "tooltip": "Toggle the whitening pass on/off."}),
                "enable_local_retouch":("BOOLEAN",{"default": False,
                                                   "tooltip": "Enable local blemish-inpainting pass (slower)."}),
                "face_score_thresh":  ("FLOAT",  {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                                                   "tooltip": "Minimum face detection confidence. Raise to skip low-confidence faces."}),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "process"
    CATEGORY      = "Image Processing"

    # ------------------------------------------------------------------
    def process(self, image, retouch_degree, whitening_degree,
                enable_whitening, enable_local_retouch, face_score_thresh):

        self._load_models()
        device = self.device

        # ComfyUI tensor: (B, H, W, C) float32 [0,1] — handle batch
        results = []
        for img_tensor in image:
            img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            # img_np is RGB at this point
            out_np = self._process_single(
                img_np, retouch_degree, whitening_degree,
                enable_whitening, enable_local_retouch, face_score_thresh,
            )
            out_tensor = torch.from_numpy(
                out_np.astype(np.float32) / 255.0
            )
            results.append(out_tensor)

        return (torch.stack(results),)

    # ------------------------------------------------------------------
    def _process_single(self, rgb_image, retouch_degree, whitening_degree,
                        enable_whitening, enable_local_retouch, face_score_thresh):
        device      = self.device
        input_size  = 512
        patch_size  = 512

        # ── 1. Skin mask (ONNX) for whitening ────────────────────────
        skin_mask = None
        if enable_whitening and whitening_degree > 0:
            small, _ = resize_on_long_side(rgb_image, 800)
            feed = {self.onnx_in[0]: small.astype("float32")}
            skin_mask = self.onnx_sess.run(self.onnx_out, input_feed=feed)[0]

        # ── 2. Face detection ─────────────────────────────────────────
        det_results = _detect_faces(self.detector, rgb_image, face_score_thresh)
        output_pred = torch.from_numpy(rgb_image).to(device)

        crop_bboxes = get_crop_bbox(det_results)
        if len(crop_bboxes) == 0:
            # No faces found — return original
            return rgb_image

        flag_bigKernal = False
        for bbox in crop_bboxes:
            roi, expand, crop_tblr = get_roi_without_padding(rgb_image, bbox)
            roi = roi_to_tensor(roi).to(device)   # BGR→RGB tensor

            if roi.shape[2] > 0.4 * rgb_image.shape[0]:
                flag_bigKernal = True

            roi = preprocess_roi(roi)

            # ── 2a. Optional local blemish pass ──────────────────────
            if enable_local_retouch:
                roi = self._retouch_local(roi, patch_size)

            # ── 2b. Main retouching U-Net ─────────────────────────────
            with torch.no_grad():
                image_resized = F.interpolate(
                    roi, (input_size, input_size), mode="bilinear", align_corners=False)
                pred_mg = self.generator(image_resized)
                pred_mg = (pred_mg - 0.5) * retouch_degree + 0.5
                pred_mg = pred_mg.clamp(0.0, 1.0)
                pred_mg = F.interpolate(pred_mg, roi.shape[2:], mode="bilinear", align_corners=False)
                pred_mg = pred_mg[0].permute(1, 2, 0)
                if len(pred_mg.shape) == 2:
                    pred_mg = pred_mg[..., None]

                pred_mg = smooth_border_mg(self.diffuse_mask, pred_mg)

                img_roi = (roi[0].permute(1, 2, 0) + 1.0) / 2.0
                pred = (1 - 2 * pred_mg) * img_roi * img_roi + 2 * pred_mg * img_roi
                pred = (pred * 255.0).byte()

            output_pred[crop_tblr[0]:crop_tblr[1],
                        crop_tblr[2]:crop_tblr[3]] = pred

        # ── 3. Whitening pass ─────────────────────────────────────────
        if enable_whitening and whitening_degree > 0 and skin_mask is not None:
            # ONNX returns float32 but numpy may upcast; force float32
            # before whiten_img sends it to MPS (no float64 support on Apple Silicon).
            skin_mask = skin_mask.astype(np.float32)
            output_pred = whiten_img(
                output_pred, skin_mask, whitening_degree,
                flag_bigKernal=flag_bigKernal)

        if not isinstance(output_pred, np.ndarray):
            output_pred = output_pred.cpu().numpy()

        return output_pred  # RGB uint8

    # ------------------------------------------------------------------
    def _retouch_local(self, image, patch_size):
        """Local blemish detection + inpainting pass."""
        device = self.device
        with torch.no_grad():
            sub_H, sub_W = image.shape[2:]

            std = F.interpolate(image, size=(768, 768), mode="bilinear", align_corners=True)
            mask = torch.sigmoid(self.detection_net(std))
            mask = F.interpolate(mask, size=(sub_H, sub_W), mode="nearest")

            hard_low  = (mask >= 0.35).float()
            hard_high = (mask >= 0.50).float()
            mask = mask * (1 - hard_high) + hard_high
            mask = mask * hard_low
            mask = 1 - mask

            sub_H_std = sub_H if sub_H % patch_size == 0 else (sub_H // patch_size + 1) * patch_size
            sub_W_std = sub_W if sub_W % patch_size == 0 else (sub_W // patch_size + 1) * patch_size

            img_pad  = F.pad(image, (0, sub_W_std - sub_W, 0, sub_H_std - sub_H), value=0)
            mask_pad = F.pad(mask,  (0, sub_W_std - sub_W, 0, sub_H_std - sub_H), value=0)

            img_patches  = patch_partition_overlap(img_pad,  p1=patch_size, p2=patch_size)
            mask_patches = patch_partition_overlap(mask_pad, p1=patch_size, p2=patch_size)

            comp_list = []
            for i in range(img_patches.shape[0]):
                img_w  = img_patches[i:i+1]
                msk_w  = mask_patches[i:i+1]
                inp    = img_w * msk_w
                out    = self.inpainting_net(inp, msk_w)
                comp   = inp + (1 - msk_w) * out
                comp_list.append(comp)

            comp = torch.cat(comp_list, dim=0)
            h_count = int(round(sub_H_std / patch_size))
            w_count = int(round(sub_W_std / patch_size))
            return patch_aggregation_overlap(comp, h=h_count, w=w_count)[:, :, :sub_H, :sub_W]


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "SkinRetouching": SkinRetouchingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkinRetouching": "Skin Retouch Node",
}
