from setuptools import setup, find_packages

setup(
    name='multigtf',
    version='1.0.0',
    author='Yarone Tokayer',
    description='A Python implementation of a multi-species gravothermal fluid code',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yaronetokayer/multigtf',
    packages=find_packages(exclude=('multigtf.dev', 'multigtf.dev.*')),
    install_requires=[
        "numpy>=2.2,<2.3",
        "scipy>=1.15",       # SciPy 1.15 works with NumPy 2.x
        "numba>=0.61.2",     # pairs with NP<=2.2
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
