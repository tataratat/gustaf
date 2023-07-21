from setuptools import setup

from docs.source.handle_markdown import process_file

with open("gustaf/_version.py") as f:
    version = eval(f.read().strip().split("=")[-1])

readme = process_file("README.md", relative_links=False, return_content=True)

setup(
    name="gustaf",
    version=version,
    description="Process and visualize numerical-analysis-geometries.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Jaewook Lee",
    author_email="jaewooklee042@gmail.com",
    url="https://github.com/tataratat/gustaf",
    packages=[
        "gustaf",
        "gustaf.utils",
        "gustaf.io",
        "gustaf.create",
        "gustaf.helpers",
    ],
    install_requires=["numpy"],
    extras_require={
        "all": [
            "vedo>=2023.4.3",
            "scipy",
            "meshio",
            "napf>=0.0.5",
            "funi>=0.0.1",
        ]
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    license="MIT",
)
