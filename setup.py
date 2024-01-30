import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

DESCRIPTION = "TensorFlow and Pytorch helpers to run experiments on Hadoop"

try:
    LONG_DESCRIPTION = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except Exception:
    LONG_DESCRIPTION = ""


def _read_reqs(relpath):
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]


REQUIREMENTS = _read_reqs("requirements.txt")

CLASSIFIERS = [
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.9",
    "Intended Audience :: Developers",
]


setuptools.setup(
    name="ml_hadoop_experiment",
    packages=setuptools.find_packages(),
    include_package_data=True,
    version="0.0.5",
    install_requires=REQUIREMENTS,
    tests_require=_read_reqs("tests-requirements.txt"),
    python_requires=">=3.6",
    maintainer="Criteo",
    maintainer_email="github@criteo.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,
    keywords="tensorflow pytorch yarn",
    url="https://github.com/criteo/ml-hadoop-experiment",
)
