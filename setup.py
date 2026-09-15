"""Setup script for the lep-data package."""

from setuptools import find_packages, setup

# The heavy scientific stack lives in the Apptainer image, not here; this list
# is what an editable install needs to import lep_data outside the container.
DEPS = [
    "awkward",
    "fastjet",
    "hydra-core",
    "jinja2",
    "numba",
    "numpy",
    "omegaconf",
    "pyarrow",
    "uproot",
    "vector",
]

setup(
    name="lep-data",
    version="0.1.0",
    description="ALEPH (LEP) ntuple production for the ML-tau project",
    author="HEP-KBFI",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "lep_data.aleph": ["config/*.yaml", "templates/*.j2"],
    },
    python_requires=">=3.8",
    install_requires=DEPS,
)
