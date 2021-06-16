import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="singan-seg-polyp", # Replace with your own username
    version="1.0.4",
    author="Vajira Thambawita",
    author_email="vlbthambawita@gmail.com",
    description="Generating synthetic polyps and corresponding mask using pretrained SinGAN-Seg and Style tranfering functionalities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vlbthambawita/singan-seg-polyp",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'singan-seg-polyp': ['config.yaml']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        #'torch',
        'numpy',
        'tqdm',
        #'torch',
        #'torchvision',
        'pandas',
        'pathlib',
        'PyYAML',
        'scipy',
        'scikit-image',
        'scikit-learn',
        'requests',
        'natsort'
        

  ],
)