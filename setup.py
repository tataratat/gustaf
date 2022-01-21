from setuptools import setup

with open("gustav/_version.py") as f:
    version = eval(f.read().strip().split("=")[-1])

setup(
    name="gustav",
    version=version,
    description="Fast geometry prototyper.",
    author="Jaewook Lee",
    author_email="jlee@ilsb.tuwien.ac.at",
    packages=[
        "gustav",
        "gustav.utils"
    ],
    install_requires=[
        "numpy",
    ],
)
