from setuptools import setup, find_packages

setup(
    name='pygtf2',
    version='1.0.0a1',
    author='Yarone Tokayer',
    description='A Python implementation of a multi-species gravothermal fluid code',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yaronetokayer/pygtf2',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.3',
        'scipy>=1.15',
        'numba>=0.61',
        'matplotlib>=3.10',
        'tqdm>=4.67'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
