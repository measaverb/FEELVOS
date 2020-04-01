from setuptools import setup, find_packages

setup(
    name            = 'feelvos',
    version         = '0.5',
    description     = 'FEELVOS implementation in PyTorch; FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation',
    author          = 'Younghan Kim',
    author_email    = 'godppkyh@mosqtech.com',
    install_requires= [],
    packages        = find_packages(),
    python_requires = '>=3.6'  
)
