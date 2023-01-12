from setuptools import setup, find_packages
import os


__version__ = 0.01

packages = find_packages(exclude=["docs", "notebooks", "assets"])

#
# Base installation (interface only)
#
install_requires = [
    "torch",
    "farmgym @ git+https://github.com/farm-gym/farm-gym",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="farm-gym-games",
    version=__version__,
    description="some farms games from farmgym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=packages,
    install_requires=install_requires,
    zip_safe=False,
    include_package_data=True,
)
