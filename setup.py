from setuptools import find_packages, setup

setup(
    name="plotly_for_scanpy",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "plotly",
        "anndata",
    ],
    entry_points={},
    description="A custom Plotly-based visualization package for Scanpy",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/plotly_for_scanpy",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
