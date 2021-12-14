from setuptools import setup

setup(
    name='vibdata',
    version='0.2',
    url='https://gitlab.com/ninfa-ufes/deep-rpdbcs/signal-datasets.git',
    license='private',
    author='Lucas Henrique Sousa Mello',
    author_email='lucashsmello@gmail.com',
    description='A DeepLearn approch to vibrational signals for bearing fault classification',
    packages=['vibdata', 'vibdata.datahandler', 'vibdata.datahandler.PU', 'vibdata.datahandler.SEU', 'vibdata.datahandler.CWRU', 'vibdata.datahandler.MFPT'],
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'tqdm',
        'sklearn', 'torchvision'
    ]

)
