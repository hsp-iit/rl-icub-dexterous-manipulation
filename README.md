<!-- SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia -->
<!-- SPDX-License-Identifier: BSD-3-Clause -->

<h1 align="center">
    iCub Dexterous Manipulation with RL
</h1>

<p align="center">
 <img src="https://github.com/hsp-iit/rl-icub-dexterous-manipulation/assets/32268209/031da1ad-ee70-4688-ac7e-8634737ae441"/>
</p>

<h4 align="center">
  RESPRECT: Speeding-up Multi-fingered Grasping with Residual Reinforcement Learning
</h4>

<div align="center">
 IEEE Robotics and Automation Letters, 2024.</div>

<div align="center">
  <a href="https://ieeexplore.ieee.org/document/10423830"><b>Paper</b></a> |
  <a href="http://arxiv.org/abs/2401.14858"><b>arXiv</b></a> |
  <a href="https://youtu.be/JRsBLVclhpg"><b>Video</b></a>
</div>

<h4 align="center">
  A Grasp Pose is All You Need: Learning Multi-fingered Grasping with Deep Reinforcement Learning from Vision and Touch
</h4>

<div align="center">
 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2023), Detroit, Michigan, USA.</div>

<div align="center">
  <a href="https://ieeexplore.ieee.org/document/10341776"><b>Paper</b></a> |
  <a href="https://arxiv.org/abs/2306.03484"><b>arXiv</b></a> |
  <a href="https://youtu.be/qc6gksKH3Mo"><b>Video</b></a>
</div>

## Table of Contents

- [Update](#updates)
- [Installation](#installation)
- [Reproduce the RESPRECT results](#reproduce-the-resprect-paper-results)
- [Reproduce the G-PAYN results](#reproduce-the-g-payn-paper-results)
- [License](#license)
- [Citing the papers](#citing-the-papers)

## Updates

2024-01-26 - Code release to replicate the results presented in the paper  **RESPRECT: Speeding-up Multi-fingered Grasping with Residual Reinforcement Learning**.

2023-07-27 - Code release to replicate the results presented in the paper  **A Grasp Pose is All You Need: Learning Multi-fingered Grasping with Deep Reinforcement Learning from Vision and Touch**.

## Installation

**Install [git-lfs](https://git-lfs.com/)** as follows.
```console
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

**Install [robotology-superbuild](https://github.com/robotology/robotology-superbuild)** as follows. For further details see the official guide for [installation from source](https://github.com/robotology/robotology-superbuild#source-installation). Note that you have to replace `FirstName`, `LastName` and `user@email.domain`.
```console
sudo apt-get install -y python3-dev python3-numpy
git config --global user.name FirstName LastName
git config --global user.email user@email.domain
git clone https://github.com/robotology/robotology-superbuild
cd robotology-superbuild
bash ./scripts/install_apt_dependencies.sh
mkdir build
cd build
cmake .. -D ROBOTOLOGY_USES_PYTHON=ON -D ROBOTOLOGY_USES_GAZEBO=OFF -D ROBOTOLOGY_ENABLE_DYNAMICS=ON
make -j8
make install
```
After the installation, add the following line to the `.bashrc` to configure your environment, replacing the `<directory-where-you-downloaded-robotology-superbuild>`.
```console
source <directory-where-you-downloaded-robotology-superbuild>/build/install/share/robotology-superbuild/setup.sh
```
**Install [superquadrics-lib](https://github.com/robotology/superquadric-lib)** as follows. 
```console
git clone --recursive https://gitlab.kitware.com/vtk/vtk.git
mkdir -p vtk/build
cd vtk/build
cmake ..
make -j8
sudo make install
cd ../..
git clone https://github.com/robotology/superquadric-lib.git
cd superquadric-lib
mkdir build && cd build
cmake -D ENABLE_BINDINGS:BOOL=ON ..
make -j8
sudo make install
```

**Clone the repository and install it**:
```console
git clone --recurse-submodules https://github.com/hsp-iit/rl-icub-dexterous-manipulation.git
cd rl-icub-dexterous-manipulation
pip install catkin_pkg && pip install -e .
```

**Download the [VGN](https://github.com/ethz-asl/vgn) and [PVR](https://github.com/sparisi/pvr_habitat) weights**:
```console
cd rl_icub_dexterous_manipulation/external/vgn
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MysYHve3ooWiLq12b58Nm8FWiFBMH-bJ' -O data.zip
unzip data
rm data.zip
cd ../../feature_extractors/moco_models
wget https://github.com/sparisi/pvr_habitat/releases/download/models/moco_croponly_l3.pth
wget https://github.com/sparisi/pvr_habitat/releases/download/models/moco_croponly_l4.pth
wget https://github.com/sparisi/pvr_habitat/releases/download/models/moco_croponly.pth
```
## Reproduce the RESPRECT paper results

To run the experiments in the paper, you can either rely on the provided ***G-PAYN*** and ***REPTILE*** models pre-trained on MSO, or retrain these models as described in the following section. For example, to train ***G-PAYN*** in the *MSO+Superquadrics* experiments, use `configs/exp_resprect/gpayn_MSO_superquadrics_MAE_save_rb.yaml` to save the replay buffer, and `configs/exp_resprect/gpayn_MSO_superquadrics_MAE.yaml` to train the model. Note that if you retrain these models, you have to modify the configuration files mentioned below accordingly.

To download the models pre-trained on *MSO*, you have to run the following:
```console
cd ../../examples/eval_dir
curl -L https://dataverse.iit.it/api/access/dataset/:persistentId/?persistentId=doi:10.48557/IBDJYT -o models.zip
unzip models.zip
rm models.zip MANIFEST.TXT
```

To train the model with ***RESPRECT***, for example in the *06_mustard_bottle+Superquadrics* experiment, you have to run the following from the `examples` directory:
```console
python3 icub_visuomanip_drl.py --cfg configs/exp_resprect/resprect_mustard_superquadrics.yaml
```

To reproduce the ***Residual*** results, you have to run the following:
```console
python3 icub_visuomanip_drl.py --cfg configs/exp_resprect/residual_mustard_superquadrics.yaml
```

To reproduce the ***Fine-Tuning*** results, you have to run the following:
```console
python3 icub_visuomanip_drl.py --cfg configs/exp_resprect/fine_tuning_mustard_superquadrics.yaml
```

To reproduce the ***Reptile*** results, you have to run the following:
```console
python3 icub_visuomanip_drl.py --cfg configs/exp_resprect/reptile_mustard_superquadrics.yaml
```

To reproduce the ***G-PAYN*** results, follow the instructions below, but consider using different configuration files to use MAE as feature extractor. For example, using `configs/exp_resprect/gpayn_mustard_superquadrics_MAE.yaml` instead of `configs/exp_gpayn/gpayn_mustard_superquadrics.yaml`.

To replicate the experiments on the real iCub humanoid, you need to run the following modules before running the provided script to train a ***RESPRECT*** policy:
- [`yarprobotinterface`](https://github.com/robotology/yarp) to run the robot.
- [`iKinCartesianSolver`](https://github.com/robotology/icub-main/tree/master) for the right arm of the robot, using the parameter`--part right_arm`.
- [`iKinGazeCtrl`](https://github.com/robotology/icub-main/tree/master), with the the parameter `--from config_no_imu_no_tilt.ini`.
- [`yarprobotinterface`](https://github.com/robotology/yarp) with the parameter `--config realsense_d405.xml`.
- [`realsense-holder-publisher`](https://github.com/robotology/realsense-holder-calibration) with the parameter `--from config_half_tilted_v27_d405.ini`.
- [`skinManager`](https://github.com/robotology/icub-main/tree/master) with the parameters `--from skinManAll.ini --context skinGui`.

Then, from the machine where you are running the experiment, you need to run `yarprun --server /laptop` and `yarpmanager`. From a separate terminal you can communicate with the robot via RPC with the command `yarp rpc --client /cmd`, for example to terminate an unsuccessful episode with the command `done_moved`. Then, in the `yarpmanager`, you need to load the application `rl_icub_dexterous_manipulation/yarp_modules/applications/app_resprect_icub.xml`, run all the modules and make all the connections.

Finally, to train the ***RESPRECT*** model for example with the *06_mustard_bottle*, you have to run the following:
```console
python3 icub_visuomanip_drl_real.py --cfg configs/exp_resprect/resprect_real_mustard.yaml
```

Note that, to reproduce the experiments on a different setting from the one shown in the paper, you may need to set the parameters for point cloud filtering differently from the ones in the provided configuration files.

## Reproduce the G-PAYN paper results

To run the experiments in the paper, you have to extract the replay buffers, unless you want to run the ***SAC*** experiments. For example, for the *06_mustard_bottle+Superquadrics* experiment, you have to run the following from the `examples` directory:

```console
python3 icub_visuomanip_drl.py --cfg configs/exp_gpayn/gpayn_mustard_superquadrics_save_rb.yaml
```

Then, to train the model with ***G-PAYN***, you have to run the following:
```console
python3 icub_visuomanip_drl.py --cfg configs/exp_gpayn/gpayn_mustard_superquadrics.yaml
```

To reproduce the ***OERLD*** results, you have to run the following:
```console
python3 icub_visuomanip_drl.py --cfg configs/exp_gpayn/oerld_mustard_superquadrics.yaml
```

To reproduce the ***AWAC*** results, you have to run the following:
```console
python3 icub_visuomanip_drl.py --cfg configs/exp_gpayn/awac_mustard_superquadrics.yaml
```

To reproduce the ***SAC*** results, you have to run the following:
```console
python3 icub_visuomanip_drl.py --cfg configs/exp_gpayn/sac_mustard_superquadrics.yaml
```

## License

The code is released under the *BSD 3-Clause License*. See [LICENCE](https://github.com/hsp-iit/rl-icub-dexterous-manipulation/blob/main/LICENSE) for further details.

## Citing the papers

If you find any part of this code useful, please consider citing the associated publications:

```bibtex
@ARTICLE{ceola2024resprect,
  author={Ceola, Federico and Rosasco, Lorenzo and Natale, Lorenzo},
  journal={IEEE Robotics and Automation Letters}, 
  title={RESPRECT: Speeding-up Multi-fingered Grasping with Residual Reinforcement Learning}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  keywords={Dexterous Manipulation; Multifingered Hands; Reinforcement Learning.},
  doi={10.1109/LRA.2024.3363532}}

@INPROCEEDINGS{ceola2023gpayn,
  author={Ceola, Federico and Maiettini, Elisa and Rosasco, Lorenzo and Natale, Lorenzo},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={A Grasp Pose is All You Need: Learning Multi-Fingered Grasping with Deep Reinforcement Learning from Vision and Touch}, 
  year={2023},
  volume={},
  number={},
  pages={2985-2992},
  doi={10.1109/IROS55552.2023.10341776}}
```

## Maintainer

This repository is maintained by:

| | |
|:---:|:---:|
| [<img src="https://github.com/hsp-iit/rl-icub-dexterous-manipulation/assets/32268209/1cc77fb9-f7f1-4b9b-9064-25584d69e57c" width="40">](https://github.com/hsp-iit) | [@fedeceola](https://github.com/fedeceola) |
