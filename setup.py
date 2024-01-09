
setup(
    name='customgbm',
    version='1.0',
    packages=find_packages(exclude=["notebooks", "data"]),
    include_package_data=True,
    install_requires=[
    'numpy==1.24.3',
    'jupyter',
    'ipykernel',
    'matplotlib==3.6.0',
    'pandas==2.0.2',
    'scikit-learn==1.2.2',
    'scipy==1.10.1',
    'lightgbm==3.3.5',
    'jax==0.4.12',
    'jaxlib=0.4.12'
    ],
    python_requires='==3.8.*',
    author='Davide Boldini',
    author_email='davide.boldini@tum.de',
    description='JAX-based implementation of custom loss functions for LightGBM',
    long_description=open('README_PyPI.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dahvida/Custom_gbm',
    # Other parameters as needed
)
