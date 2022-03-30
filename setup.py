from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf8") as fh:
    LONG_DESCRIPTION = fh.read()

REQUIREMENTS = [
    "configsuite>=0.6",
    "cwrap~=1.6",
    "opm==2020.10.2",
    "ecl~=2.13.1",
    "ecl2df~=0.16.1",
    "ert~=2.19.0",
    "fmu-ensemble~=1.5.0",
    "hyperopt~=0.2.5",
    "matplotlib~=3.1",
    "mlflow>=1.11.0",
    "numpy~=1.21.5",
    "pandas~=1.0",
    "psutil~=5.7",
    "pykrige~=1.5",
    "pyvista~=0.23",
    "pyyaml~=5.2",
    "scikit-learn~=0.22",
    "scipy~=1.6",
    "webviz-config~=0.3.8",
    "webviz-config-equinor~=0.2.2",
    "webviz-subsurface~=0.2.9",
    "xlrd<2",
]

TEST_REQUIRES = [
    "black",
    "mypy>=0.761",
    "pylint>=2.3",
    "pyscal>=0.7.4",
    "pytest>=5.3",
    "pytest-cov>=2.8",
    "sphinx",
    "sphinx-rtd-theme",
    "types-pkg_resources",
    "types-PyYAML",
    "pre-commit~=2.9.3",
]

setup(
    name="flownet",
    install_requires=REQUIREMENTS,
    tests_require=TEST_REQUIRES,
    setup_requires=["setuptools_scm~=3.2"],
    python_requires="~=3.7",
    extras_require={"tests": TEST_REQUIRES},
    description="Simplified training of reservoir simulation models",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/equinor/flownet",
    author="R&T Equinor",
    use_scm_version=True,
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={
        "flownet": ["templates/*", "static/*", "ert/forward_models/FLOW_SIMULATION"]
    },
    entry_points={
        "ert": ["flow = flownet.ert.forward_models._flow_job"],
        "console_scripts": [
            "flownet=flownet._command_line:main",
            "flownet_render_realization=flownet.ert.forward_models:render_realization",
            "flownet_delete_simulation_output=flownet.ert.forward_models:delete_simulation_output",
            "flownet_run_flow=flownet.ert.forward_models:run_flow",
            "flownet_save_iteration_parameters=flownet.ert.forward_models:save_iteration_parameters",
            "flownet_save_iteration_analytics=flownet.ert.forward_models:save_iteration_analytics",
            "flownet_save_predictions=flownet.ert.forward_models:save_predictions",
            "flownet_plot_results=flownet.utils.plot_results:main",
        ],
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
