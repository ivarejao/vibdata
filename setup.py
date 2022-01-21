from setuptools import setup

setup(
    name='vibdata',
    version='0.4.5',
    url='https://gitlab.com/ninfa-ufes/deep-rpdbcs/signal-datasets.git',
    license='private',
    author='Lucas Henrique Sousa Mello',
    author_email='lucashsmello@gmail.com',
    description='A library for loading vibration signals datasets',
    packages=['vibdata', 'vibdata.datahandler', 'vibdata.datahandler.PU', 'vibdata.datahandler.SEU', 'vibdata.datahandler.CWRU',
     'vibdata.datahandler.MFPT', 'vibdata.datahandler.RPDBCS', 'vibdata.datahandler.transforms'],
    package_data={'vibdata.datahandler.PU': ['PU.csv'], 'vibdata.datahandler.SEU': ['SEU.csv'],
                  'vibdata.datahandler.CWRU': ['CWRU.csv'], 'vibdata.datahandler.MFPT': ['MFPT.csv']},
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'tqdm',
        'scikit-learn', 'torchvision'
    ]

)
