from setuptools import setup, find_packages

setup(
    name="matepoint",
    version="0.1.2",
    packages=find_packages(),
    install_requires=["torch>=2.4.0"],
    author="Windborne Systems",
    author_email="anuj@windbornesystems.com",
    description="A fork of PyTorch's checkpoint that trades CPU RAM for effective GPU VRAM",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/windbornesystems/matepoint",
    python_requires=">=3.10",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",  
        "Programming Language :: Python :: 3",
    ],
)
