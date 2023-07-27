# SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Based on code from https://github.com/sparisi/pvr_habitat.git with CC BY-NC 4.0 licence

import torch
import numpy as np
from rl_icub_dexterous_manipulation.external.pvr_habitat.src.vision_models.moco import \
    moco_conv3_compressed, moco_conv4_compressed, moco_conv5
import torchvision.transforms as T


class ImagesFeatureExtractorMOCO:
    # https://github.com/sparisi/pvr_habitat
    def __init__(self,
                 model_name='moco_croponly',
                 device='cuda'):

        self.device = device
        self.model = EmbeddingNet(embedding_name=model_name)

        # Compute the features dimension
        fake_im_array = np.random.rand(480, 640, 3) * 255
        self.output_features_dimension = self.__call__(fake_im_array).shape

    def __call__(self, image):
        image_input = torch.from_numpy(image.copy()/255).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model(image_input)
        return image_features


class EmbeddingNet(torch.nn.Module):
    """
    Input shape must be (N, H, W, 3), where N is the number of frames.
    The class will then take care of transforming and normalizing frames.
    The output shape will be (N, O), where O is the embedding size.
    """
    def __init__(self, embedding_name, in_channels=3):
        super(EmbeddingNet, self).__init__()
        self.embedding_name = embedding_name

        self.in_channels = in_channels
        if embedding_name == "moco_croponly_l3":
            self.embedding = moco_conv3_compressed(
                checkpoint_path='../feature_extractors/moco_models/moco_croponly_l3.pth')
        elif embedding_name == "moco_croponly_l4":
            self.embedding = moco_conv4_compressed(
                checkpoint_path='../feature_extractors/moco_models/moco_croponly_l4.pth')
        elif embedding_name == "moco_croponly":
            self.embedding = moco_conv5(checkpoint_path='../feature_extractors/moco_models/moco_croponly.pth')

        # Set the model in evaluation mode
        self.embedding.eval()
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.transforms = torch.nn.Sequential(
            T.Resize(256),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        dummy_in = torch.zeros(1, in_channels, 480, 640)
        dummy_in = self.transforms(dummy_in)
        self.in_shape = dummy_in.shape[1:]
        dummy_out = self._forward(dummy_in)
        self.out_size = np.prod(dummy_out.shape)

        # Always use CUDA, it is much faster for these models
        # Disable it only for debugging
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.embedding = self.embedding.to(device=self.device)
        self.training = self.embedding.training

    def _forward(self, observation):
        return self.embedding(observation)

    def forward(self, observation):
        # observation.shape -> (N, H, W, 3)
        observation = observation.to(device=self.device)
        observation = observation.transpose(1, 2).transpose(1, 3).contiguous()
        observation = self.transforms(observation)
        observation = observation.reshape(-1, *self.in_shape)

        with torch.no_grad():
            out = self._forward(observation)
            return out.view(1, self.out_size).cpu()
