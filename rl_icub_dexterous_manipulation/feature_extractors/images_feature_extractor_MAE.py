# SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Based on code from https://github.com/ir413/mvp.git

import torch
import mvp
import cv2


class ImagesFeatureExtractorMAE:

    def __init__(self,
                 model_name,
                 device='cuda'):

        self.device = device
        self.model_name = model_name
        self.model = mvp.load("vitl-256-mae-egosoup")
        self.model.to(self.device)
        self.model.freeze()
        self.output_features_dimension = (1, 1024)
        self.mean_im = [0.485, 0.456, 0.406]
        self.std_im = [0.229, 0.224, 0.225]

    def __call__(self, image):
        # Consider rendering the image from MuJoCo already with size (256, 256)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        # copy() needed to avoid ValueError: At least one stride in the given numpy array is negative, and tensors with
        # negative strides are not currently supported.
        image_input = torch.from_numpy(image.copy()).to(self.device).float()
        image_input /= 255.0
        for i in range(3):
            image_input[:, :, i] = (image_input[:, :, i] - self.mean_im[i]) / self.std_im[i]
        image_input = image_input.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            image_features = self.model.extract_feat(image_input)
        return image_features.cpu()
