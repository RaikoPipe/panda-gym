import os

from setuptools import find_packages, setup

with open(os.path.join("panda_gym", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="panda_gym",
    description="Set of robotic environments based on PyBullet physics engine and gymnasium. "
                "Fork of panda-gym by https://github.com/qgallouedec",
    author="Richard Reider",
    author_email="richard.reider@ovgu.de",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RaikoPipe/panda-gym",
    packages=find_packages(),
    include_package_data=True,
    package_data={"panda_gym": ["version.txt"]},
    version=__version__,
    install_requires=["gymnasium~=0.29.1", "pybullet", "numpy", "scipy", "torch", "wandb", "seaborn",
                      "pyb_utils @ git+https://git@github.com/RaikoPipe/pyb_utils",
                      "stable_baselines3 @ git+https://git@github.com/RaikoPipe/stable-baselines3",
                      "sb3_extensions @ git+https://git@github.com/RaikoPipe/sb3_extensions",
                      "sb3-contrib",
                      "progress==1.6",
                      "roboticstoolbox-python==1.1.0",
                      "tensorboard"],
    dependency_links=["https://download.pytorch.org/whl/cu124"],
    extras_require={
        "develop": ["pytest-cov", "black", "isort", "pytype", "sphinx", "sphinx-rtd-theme"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11"
    ],
)
