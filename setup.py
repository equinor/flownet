from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()


REQUIREMENTS = [
    "configsuite>=0.6",
    "cwrap>=1.6",
    "ecl2df>=0.5",
    "fmu-ensemble>=1.2",
    "jinja2>=2.10",
    "matplotlib>=3.1",
    "numpy>=1.17",
    "opm>=2020.10",
    "pandas~=1.0",
    "pyarrow>=0.14",
    "pyscal>=0.4.1",
    "pyvista>=0.23",
    "pyyaml>=5.2",
    "scikit-learn>=0.22",
    "scipy>=1.4",
    "webviz-config>=0.0.42",
    "webviz-config-equinor>=0.0.9",
    "webviz-subsurface>=0.0.24",
]

TEST_REQUIRES = [
    "black",
    "mypy>=0.761",
    "pylint>=2.3",
    "pytest>=5.3",
    "pytest-cov>=2.8",
    "sphinx",
    "sphinx-rtd-theme",
]

setup(
    name="flownet",
    install_requires=REQUIREMENTS,
    tests_require=TEST_REQUIRES,
    setup_requires=["setuptools_scm~=3.2"],
    python_requires="~=3.6",
    extras_require={"tests": TEST_REQUIRES},
    description="Simplified training of reservoir simulation models",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/equinor/flownet",
    author="R&T Equinor",
    use_scm_version=True,
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"flownet": ["templates/*", "static/*"]},
    entry_points={
        "console_scripts": [
            "flownet=flownet._command_line:main",
            "flownet_render_realization=flownet.realization:render_realization",
            "flownet_delete_simulation_output=flownet.ahm:delete_simulation_output",
            "flownet_save_iteration_parameters=flownet.ahm:save_iteration_parameters",
            "flownet_save_iteration_analytics=flownet.ahm:save_iteration_analytics",
        ]
    },
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
