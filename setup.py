from setuptools import setup, find_packages

setup(
    name='icub_mujoco',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'imitation==0.3.1',
        'dm_control==1.0.8',
        'stable-baselines3[extra]==1.5.0',
        'pyyaml',
        'torchvision',
        'pyquaternion',
        'clip @ git+https://github.com/openai/CLIP.git',
        'd3rlpy==1.1.1'
    ]
)
