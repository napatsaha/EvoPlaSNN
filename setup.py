from setuptools import setup, find_packages


setup(
    name="snn_test",
    version="0.1.0",
    author="Napat Sahapat",
    description="Simple Spiking Neural Network (SNN) library with learnable synapses",
    packages=find_packages(include=["snn", "evo"], where="src"),
    package_dir={"": "src"}
)