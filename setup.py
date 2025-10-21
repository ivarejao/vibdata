from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='vibdata',
    version='1.1.3',
    description='A package for vibration signal datasets',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ivarejao/vibdata/tree/signal_baseline',
    author='Vitor Boenlla',
    packages=find_packages(),
    install_requires=["torch",
                      "tqdm",
                      "numpy",
                      "requests",
                      "pandas",
                      "scikit-learn",
                      "rarfile",
                      "scipy",
                      "opencv-python",
                      "gdown",
                      "essentia"],
    include_package_data=True,
    package_data={'': ['**/*.csv']}
)