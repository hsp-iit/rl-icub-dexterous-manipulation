

<h1 align="center">
    iCub Dexterous Manipulation with RL
</h1>

<p align="center">
 <img src=(https://github.com/hsp-iit/rl-icub-dexterous-manipulation/assets/32268209/031da1ad-ee70-4688-ac7e-8634737ae441)/>
</p>

<h4 align="center">
  A Grasp Pose is All You Need: Learning Multi-fingered Grasping with Deep Reinforcement Learning from Vision and Touch
</h4>

<div align="center">
 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2023), Detroit, Michigan, USA.</div>

<div align="center">
  <a href=""><b>Paper</b></a> |
  <a href="https://arxiv.org/abs/2306.03484"><b>arXiv</b></a> |
  <a href="https://youtu.be/qc6gksKH3Mo"><b>Video</b></a> |
</div>

## Table of Contents

- [Update](#updates)
- [Installation](#installation)
- [Reproduce the results](#reproduce-the-paper-results)
- [License](#license)
- [Citing this paper](#citing-this-paper)

## Updates

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

## Reproduce the paper results

To run the experiments in the paper, you have to extract the replay buffers, unless you want to run the ***SAC*** experiments. For example, for the *06_mustard_bottle+Superquadrics* experiment, you have to run the following from the `examples` directory:

```console
python3 icub_visuomanip_drl.py --cfg configs/gpayn_mustard_superquadrics_save_rb.yaml
```

Then, to train the model with ***G-PAYN***, you have to run the following:
```console
python3 icub_visuomanip_drl.py --cfg configs/gpayn_mustard_superquadrics.yaml
```

To reproduce the ***OERLD*** results, you have to run the following:
```console
python3 icub_visuomanip_drl.py --cfg configs/oerld_mustard_superquadrics.yaml
```

To reproduce the ***AWAC*** results, you have to run the following:
```console
python3 icub_visuomanip_drl.py --cfg configs/awac_mustard_superquadrics.yaml
```

To reproduce the ***SAC*** results, you have to run the following:
```console
python3 icub_visuomanip_drl.py --cfg configs/sac_mustard_superquadrics.yaml
```

## License

The code is released under the *BSD 3-Clause License*. See [LICENCE](https://github.com/hsp-iit/rl-icub-dexterous-manipulation/blob/main/LICENSE) for further details.

## Citing this paper

If you find any part of this code useful, please consider citing the associated publication:

```bibtex
@article{ceola2023grasp,
  title={A Grasp Pose is All You Need: Learning Multi-fingered Grasping with Deep Reinforcement Learning from Vision and Touch},
  author={Ceola, Federico and Maiettini, Elisa and Rosasco, Lorenzo and Natale, Lorenzo},
  journal={arXiv preprint arXiv:2306.03484},
  year={2023}
}
```

## Maintainer

This repository is maintained by:

| | |
|:---:|:---:|
| [<img src="https://github.com/hsp-iit/rl-icub-dexterous-manipulation/assets/32268209/1cc77fb9-f7f1-4b9b-9064-25584d69e57c" width="40">](https://github.com/hsp-iit) | [@fedeceola](https://github.com/fedeceola) |
