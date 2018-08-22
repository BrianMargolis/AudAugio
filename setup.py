import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AudAugio",
    version="0.0.1a1",
    author="Brian Margolis",
    author_email="BrianMargolis2019@u.northwestern.edu",
    description="Augments audio for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BrianMargolis/AudAugio",
    packages=setuptools.find_packages(),
    install_requires=['librosa', 'numpy', 'sox'],
    python_requires='>=3.0',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
