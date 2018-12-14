import io
from setuptools import setup, find_packages

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='HypEval',
    version='0.1.0',
    packages=find_packages(exclude=['examples']),
    license='',
    long_description=readme,
    install_requires=['numpy', 'scikit-learn']
)
