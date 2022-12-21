from setuptools import setup

with open("gustaf/_version.py") as f:
    version = eval(f.read().strip().split("=")[-1])

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="gustaf",
    version=version,
    description="Process and visualize geometries.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Jaewook Lee",
    author_email="jaewooklee042@gmail.com",
    url="https://github.com/tataratat/gustaf",
    packages=[
        "gustaf",
        "gustaf.utils",
        "gustaf.io",
        "gustaf.spline",
        "gustaf.create",
        "gustaf.helpers",
        "gustaf.spline.microstructure",
    ],
    install_requires=[
        "numpy",
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'
    ],
    license="MIT",
)
