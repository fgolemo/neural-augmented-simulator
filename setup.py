from setuptools import setup

setup(
    name='nsm',
    version='1.0',
    install_requires=[
        "torch>=1.0", "tqdm", "numpy", "matplotlib", 'gym>=0.10',
        'mujoco_py>=1.15'
    ])
