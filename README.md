<p align="center">
  <img height="175" src="https://raw.githubusercontent.com/equinor/flownet/master/docs/_static/flownet_logo.svg">
</p>

<h2 align="center">FlowNet: Data-Driven Reservoir Predictions</h2>

<p align="center">
<a href="https://pypi.org/project/flownet/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/flownet"></a>
<a href="https://github.com/equinor/flownet/actions/workflows/flownet.yml"><img alt="CI status" src="https://img.shields.io/github/actions/workflow/status/equinor/flownet/flownet.yml?branch=master&amp;label=CI"></a>
<a href="https://www.python.org/"><img alt="Python 3.12 and 3.14 tested" src="https://img.shields.io/badge/python-3.12%20%7C%203.14-blue.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: Black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/equinor/flownet/blob/master/LICENSE"><img alt="License: GPLv3" src="https://img.shields.io/github/license/equinor/flownet"></a>
</p>
<br/>

_FlowNet_ aims at solving the following problems:

* Create data-driven reduced-physics models directly from data
* Train the models
* Assess model predictiveness
* Use the models to optimize and make decisions efficiently

<p align="center">
  <img height="150" src="https://raw.githubusercontent.com/equinor/flownet/master/docs/_static/flownet_model.svg">
</p>

See the [FlowNet documentation](https://equinor.github.io/flownet/).

## Contributing

Please check out our [contribution guidelines](CONTRIBUTING.md) if you want to contribute to FlowNet.

## Installation

_FlowNet_ is a Python package. Its Python dependencies are installed with the package,
but the [_OPM Flow_](https://opm-project.org/?page_id=19) simulator binary must be
installed separately.

If OPM Flow is not installed at `/usr/bin/flow`, set the `FLOW_PATH` environment
variable to the path of the Flow executable before running FlowNet.

### Install FlowNet

The recommended approach is to install FlowNet from PyPI:
```bash
python -m pip install flownet
```

To install the latest unreleased code:
```bash
git clone https://github.com/equinor/flownet.git
cd flownet
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```
Omit the `-e` flag if you want a standard installation.

Python 3.12 and 3.14 are tested on Ubuntu 24.04 and 26.04, respectively. Python
3.13 is within the declared dependency range but is not a separate CI target. To
install test dependencies, run `python -m pip install -e '.[tests]'` from the
repository root.

> **Note:** When using the LSF queue, make sure the shell used by ERT activates the
> virtual environment. You may need to source it from your `.cshrc` or `.bashrc`.

### Running FlowNet

You can run _FlowNet_ as a single command line:
```
flownet ahm ./some_config.yaml ./some_output_folder
```
Run `flownet --help` to see all possible command line argument options.

### Running webviz to check results

Before running `webviz` for the first time, create and install a localhost HTTPS
certificate:
```bash
webviz certificate --auto-install --force
```

### License

FlowNet is, with the data and logo exceptions listed below, licensed under [GPLv3](./LICENSE).

- The Norne input model, distributed in the [FlowNet test-data repository](https://github.com/equinor/flownet-testdata), is available under the [Open Database License](https://opendatacommons.org/licenses/odbl/1.0/).
- The Egg input model, distributed in the [FlowNet test-data repository](https://github.com/equinor/flownet-testdata), is subject to [4TU.ResearchData's general terms of use](https://data.4tu.nl/article/online_resource/General_terms_of_use_for_4TU_Centre_for_Research_Data/12721292).
- The [FlowNet logo](docs/_static/flownet_logo.svg) is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
