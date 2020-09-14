import setuptools


with open("README.md", "r") as rdr:
    long_description = rdr.read()


setuptools.setup(
    name="active_learning",
    version="0.1.0",
    author="Logan Ward",
    author_email="lward@anl.gov",
    description="Active learning library for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/globus-labs/active-learning",
    packages=setuptools.find_packages(),
    install_requires=[
        'scikit-learn'
    ],
    python_requires='>3.5',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
