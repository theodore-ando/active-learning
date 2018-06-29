import setuptools


with open("README.md", "r") as rdr:
    long_description = rdr.read()


setuptools.setup(
    name="active-learning",
    version="0.0.1",
    author="Theodore Ando",
    author_email="tando2@icloud.com",
    description="Active learning library for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theodore-ando/active-learning",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)