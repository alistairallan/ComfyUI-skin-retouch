import os
import torch
import numpy as np
from PIL import Image
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class SkinRetouchingNode:
    def __init__(self):
        # Initialize the model upon node creation
        self.skin_retouching = pipeline(Tasks.skin_retouching, model='damo/cv_unet_skin-retouching')

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Image Processing"

    def process(self, image):
        # Convert tensor to a NumPy array readable by OpenCV
        # The input tensor is in the format (batch_size, height, width, channels)
        # and has values in the range [0, 1].
        image_np = 255. * image.cpu().numpy().squeeze()
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        # Convert from RGB to BGR for OpenCV
        image_bgr = image_np[:, :, ::-1]

        # Apply skin retouching
        processed_image_bgr = self.skin_retouching(image_bgr)[OutputKeys.OUTPUT_IMG]

        # Convert back from BGR to RGB
        processed_image_rgb = processed_image_bgr[:, :, ::-1]

        # Convert back to a tensor for ComfyUI
        processed_image_tensor = torch.from_numpy(processed_image_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (processed_image_tensor,)

# A dictionary that ComfyUI uses to register the nodes
NODE_CLASS_MAPPINGS = {
    "SkinRetouching": SkinRetouchingNode
}

# A dictionary that allows the user to display a more human-readable name in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SkinRetouching": "Skin Retouch Node"
}