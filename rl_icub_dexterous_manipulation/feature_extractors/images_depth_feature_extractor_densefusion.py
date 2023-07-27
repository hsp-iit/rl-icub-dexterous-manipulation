# Code adapted from https://github.com/j96w/DenseFusion, mainly from
# https://github.com/j96w/DenseFusion/blob/master/tools/eval_ycb.py
import sys
sys.path.insert(0, '../external/DenseFusion')
import numpy as np
import numpy.ma as ma
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from rl_icub_dexterous_manipulation.external.DenseFusion.lib.network import PoseNetFeat
import rl_icub_dexterous_manipulation.external.DenseFusion.lib.extractors as extractors
from torch.nn import functional as F


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class PSPNet(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNet, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax(dim=-1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)

        return self.final(p)


class ModifiedResnet(nn.Module):

    def __init__(self):
        super(ModifiedResnet, self).__init__()

        self.model = lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18')
        self.model = nn.DataParallel(self.model())

    def forward(self, x):
        x = self.model(x)
        return x


class PoseNetFeatureExtractor(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeatureExtractor, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)

        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, 21 * 4, 1)  # quaternion
        self.conv4_t = torch.nn.Conv1d(128, 21 * 3, 1)  # translation
        self.conv4_c = torch.nn.Conv1d(128, 21 * 1, 1)  # confidence

        self.num_obj = 21

    def forward(self, img, x, choose):
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        return ap_x


class ImagesDepthFeatureExtractorDenseFusion:

    def __init__(self):
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.cam_cx = 320
        self.cam_cy = 240
        self.cam_fx = 617.783447265625
        self.cam_fy = 617.783447265625
        self.cam_scale = 1.0
        self.num_points = 1000

        self.estimator = PoseNetFeatureExtractor(num_points=self.num_points)
        self.estimator.cuda()
        self.estimator.load_state_dict(torch.load('../external/DenseFusion/trained_checkpoints/ycb/'
                                                  'pose_model_26_0.012863246640872631.pth'))
        self.estimator.eval()
        self.output_features_dimension = (1, 470)

    def __call__(self, img, depth, mask, obj_segm_id):
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = mask[:, :, 0] == obj_segm_id
        mask = mask_depth * mask_label
        ids = np.where(mask_label == 1)
        if len(ids[0]) <= 1:
            return torch.zeros((1, 470), dtype=torch.float32)
        rmin = min(ids[0])
        rmax = max(ids[0])
        cmin = min(ids[1])
        cmax = max(ids[1])
        if rmin == rmax or cmin == cmax:
            return torch.zeros((1, 470), dtype=torch.float32)
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        elif len(choose) == 0:
            return torch.zeros((1, 470), dtype=torch.float32)
        else:
            choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        pt2 = depth_masked / self.cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        img_masked = np.array(img)[:, :, :3]
        img_masked = np.transpose(img_masked, (2, 0, 1))
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        cloud = torch.from_numpy(cloud.astype(np.float32))
        choose = torch.LongTensor(choose.astype(np.int32))
        # https://github.com/j96w/DenseFusion/issues/39
        img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))
        cloud = Variable(cloud).cuda()
        choose = Variable(choose).cuda()
        img_masked = Variable(img_masked).cuda()
        cloud = cloud.view(1, self.num_points, 3)
        img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])
        with torch.no_grad():
            features = self.estimator(img_masked, cloud, choose)

        # Added some mean operations to reduce features dimensionality
        features_padded = torch.zeros(1410, device=features.device)
        features_padded[:1408] = torch.mean(features.squeeze(), dim=1)
        features = torch.mean(features_padded.view(470, 3), dim=1).unsqueeze(0)

        return features.cpu()


