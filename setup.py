#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
    name="alepython",
    description="Python Accumulated Local Effects (ALE) package.",
    author="Maxime Jumelle",
    author_email="maxime@blent.ai",
    license="Apache 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaximeJumelle/alepython/",
    install_requires=required,
    setup_requires=["setuptools-scm"],
    use_scm_version=dict(write_to="src/alepython/_version.py"),
    keywords="alepython",
    package_dir={"": "src"},
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approched :: Apache 2",
        "Operating System :: OS Independent",
    ],
)
