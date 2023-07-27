import torch
import clip
from PIL import Image
import numpy as np


class ImagesFeatureExtractorCLIP:

    # https://github.com/openai/CLIP
    def __init__(self,
                 model_name,
                 device='cuda'):

        self.device = device
        self.model_name = model_name
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # Compute the features dimension
        fake_im_array = np.random.rand(480, 640, 3) * 255
        self.output_features_dimension = self.__call__(np.array(fake_im_array, dtype=np.uint8)).numpy().shape

    def __call__(self, image):
        image_input = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        return image_features.cpu()
