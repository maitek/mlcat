#!/usr/bin/env python3
from setuptools import setup, find_packages
import codecs

package_name = "mlcat"
version = 0.1

setup(
    name=package_name,
    version=version,
    description="Utilities and helpers machine learning development",
    packages=["mlcat"],
    package_dir={"mlcat": "src/mlcat"},
    install_requires=["matplotlib"]
)
