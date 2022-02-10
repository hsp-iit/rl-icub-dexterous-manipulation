# icub_mujoco

Installation guide:

Install [**MuJoCo**](https://github.com/deepmind/mujoco/)

```
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
tar -xf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

Install **GLEW**
```
sudo apt install libglew-dev
```

**Clone** the repository, **download meshes** and **install** it:
```
git clone https://github.com/fedeceola/icub_mujoco.git
cd icub_mujoco/icub_mujoco/meshes/iCub
wget --content-disposition https://istitutoitalianotecnologia-my.sharepoint.com/:u:/g/personal/federico_ceola_iit_it/EYmkAiOMA5RHkuA-flnYeeUB9J52xdH1e-bBeN7cZcPrAg?download=1
unzip -j meshes.zip && rm meshes.zip && cd ../../..
pip install -e .
```
