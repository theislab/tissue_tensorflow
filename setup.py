from setuptools import setup, find_packages
import versioneer

author = 'David S. Fischer'
author_email = 'david.fischer@helmholtz-muenchen.de'
description = ""

with open("README.md", "r") as fh:
     long_description = fh.read()

setup(
    name='tissue',
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'anndata',
        'numpy>=1.16.4',
        'pandas',
        'pillow',
        'scanpy',
        'scipy>=1.2.1',
        'scikit-learn',
        'tensorflow>=2.4.1',
        'umap-learn',
    ],
    extras_require={
        'plotting_deps': [
            'seaborn',
            'matplotlib',
            'networkx',
            'zipfile'
        ],
        'docs': [
            'sphinx',
            'sphinx-autodoc-typehints',
            'sphinx_rtd_theme',
            'jinja2',
            'docutils',
        ],
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
