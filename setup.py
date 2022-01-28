from setuptools import setup

with open("gustav/_version.py") as f:
    version = eval(f.read().strip().split("=")[-1])

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="gustav",
    version=version,
    description="Fast geometry prototyper.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Jaewook Lee",
    author_email="jlee@ilsb.tuwien.ac.at",
    packages=[
        "gustav",
        "gustav.utils",
        "gustav.io",
        "gustav.spline",
        "gustav.create"
    ],
    install_requires=[
        "numpy",
    ],
)
