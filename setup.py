from setuptools import find_packages, setup


with open('requirements.txt') as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith('#')
    ]

try:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = (
        'OrthoReg: Orthogonal Regularization for Hybrid Symbolic-Neural Dynamical Systems'
    )

setup(
    name='orthoreg',
    version='0.1.0',
    packages=find_packages(exclude=['tests', 'experiments', 'docs']),
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
        ],
        'logging': [
            'wandb>=0.15.0',
        ],
        'experiments': [
            'pandas>=1.5.0',
        ],
    },
    author='Anonymous Authors',
    author_email='anon@example.com',
    description='Orthogonal Regularization for Hybrid Symbolic-Neural Dynamical Systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/richtertill/OrthoReg',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.10',
    keywords=(
        'hybrid-modeling dynamical-systems symbolic-regression '
        'neural-networks orthogonal-regularization sindy'
    ),
    project_urls={
        'Bug Reports': 'https://github.com/richtertill/OrthoReg/issues',
        'Source': 'https://github.com/richtertill/OrthoReg',
    },
)
