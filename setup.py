from setuptools import setup, find_packages

setup(
    name='icub_mujoco',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'dm_control==0.0.403778684',
        'stable-baselines3[extra]==1.3.0',
        'pyyaml',
        'torchvision',
        'pyquaternion',
        'clip @ git+https://github.com/openai/CLIP.git'
    ]
)
