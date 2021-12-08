from setuptools import setup

setup(
    name='vibenet',
    version='0.2',
    url='https://gitlab.com/ninfa-ufes/deep-rpdbcs/signal-datasets.git',
    license='MIT',
    author='Lucas Henrique Sousa Mello',
    author_email='lucashsmello@gmail.com',
    description='A DeepLearn approuch to vibrational signals for bearing fault classification',
    packages=['datahandler', 'datahandler.PU', 'datahandler.SEU', 'datahandler.CWRU', 'datahandler.MFPT'],
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'importlib',
        'scipy',
        'os', 'tqdm',
        'abc', 'typing',
        'sklearn', 'torchvision'
    ]

)
