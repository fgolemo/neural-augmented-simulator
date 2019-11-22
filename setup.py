from setuptools import setup

setup(
    name='nas',
    version='1.0',
    packages=['neural_augmented_simulator'],
    install_requires=[
        "torch>=1.0", "tqdm", "numpy", "matplotlib", 'gym>=0.10',
        'mujoco_py>=1.15'
    ])
