import torch
from torchvision import transforms
from PIL import Image
import numpy as np


class ImagesFeatureExtractor:

    # https://pytorch.org/hub/pytorch_vision_alexnet/
    def __init__(self,
                 model_name,
                 device='cuda'):
        self.model_name = model_name
        self.model = torch.hub.load('pytorch/vision:v0.10.0', self.model_name, pretrained=True)
        self.model = self.model.eval()

        # Remove the last two layers of the classifier
        self.model.classifier = torch.nn.Sequential(*[self.model.classifier[i]
                                                      for i in range(len(self.model.classifier) - 2)])

        # Setting pre-processing specifics
        self.resize_dim = 256
        self.center_crop_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.preprocess = transforms.Compose([transforms.Resize(self.resize_dim),
                                              transforms.CenterCrop(self.center_crop_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=self.mean, std=self.std)])

        # Set device for inference
        self.device = device
        if self.device == 'cuda':
            self.model.to('cuda')

        # Compute the features dimension
        fake_im_array = np.random.rand(480, 640, 3) * 255
        self.output_features_dimension = self.__call__(np.array(fake_im_array, dtype=np.uint8)).numpy().shape

    def __call__(self, image):
        image = Image.fromarray(image)
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        if self.device == 'cuda':
            input_batch = input_batch.to('cuda')
        with torch.no_grad():
            output = self.model(input_batch)
        return output.cpu()
