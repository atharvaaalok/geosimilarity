from setuptools import find_packages, setup


# Read contents of readme.md to use as long description later
with open('README.md', 'r', encoding = 'utf-8') as f:
    readme = f.read()


setup(
    name = 'geosimilarity',
    author = 'Atharva Aalok',
    author_email = 'atharvaaalok@gmail.com',
    version = '0.0.0',
    url = 'https://github.com/atharvaaalok/geosimilarity',
    license = 'MIT',
    description = 'Differentiable curve and surface similarity measures.',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    packages = find_packages(),
    install_requires = [
        'matplotlib==3.10.0',
        'numpy==2.2.3',
        'setuptools==57.4.0',
        'svg.path==6.3',
        'torch==2.5.1',
        'typing_extensions==4.12.2'
    ],
    keywords = ['geosimilarity', 'geometry', 'neural networks', 'loss functions',
                'machine learning', 'deep learning', 'optimization', 'curve fitting', 'pytorch',
                'autograd', 'similarity measures', 'surfaces', 'curves'],
)