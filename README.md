<p align="center">
  <img height="175" src="https://raw.githubusercontent.com/equinor/flownet/master/docs/_static/flownet_logo.svg">
</p>

<h2 align="center">FlowNet: Data-Driven Reservoir Predictions</h2>

<p align="center">
<a href="https://badge.fury.io/py/flownet"><img src="https://badge.fury.io/py/flownet.svg"></a>
<a href="https://github.com/equinor/flownet/actions?query=workflow%3ACI"><img src="https://img.shields.io/github/workflow/status/equinor/flownet/CI"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.6%20|%203.7-blue.svg"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://lgtm.com/projects/g/equinor/flownet/alerts/"><img alt="Total alerts" src="https://img.shields.io/lgtm/alerts/g/equinor/flownet.svg?logo=lgtm&logoWidth=18"/></a>
<a href="https://lgtm.com/projects/g/equinor/flownet/context:python"><img src="https://img.shields.io/lgtm/grade/python/g/equinor/flownet.svg?logo=lgtm&logoWidth=18"></a>
</p>
<br/>

_FlowNet_ aims at solving the following problems:

* Create data-driven reduced physics models - directly from the data
* Train the model
* Assure model predictiveness
* Use the models to efficiently optimize and make decisions

<p align="center">
  <img height="150" src="https://raw.githubusercontent.com/equinor/flownet/master/docs/_static/flownet_model.svg">
</p>

For documentation, see [the GitHub pages](https://equinor.github.io/flownet/) for this repository.

## Contributing

Please check out our [contribution guidelines](CONTRIBUTING.md) if you want to contribute to FlowNet.

## Installation

_FlowNet_ is a Python package. The package itself, and other dependencies,
can be installed using a Python virtual environment, except for the [_OPM-Flow_](https://opm-project.org/?page_id=19) reservoir
simulator.

### Install FlowNet

_FlowNet_ uses the open-source reservoir simulator _OPM-Flow_. To be able to run _FlowNet_ you will need to have _OPM-Flow_
installed first. There are also other dependencies like the Python packages [`libecl`](https://github.com/equinor/libecl) and
[`libres`](https://github.com/equinor/libres) which currently are not easily installable from PyPI (however, things are happening, so hopefully in a not too distant future, dependencies are installable from PyPI, which is already the case for flownet itself: `pip install flownet`).

##### 1. Clone the _FlowNet_ GitHub repository with SSH:
    
```bash
git clone git@github.com:equinor/flownet.git
```

##### 2. Move into the cloned directory:
```bash
cd flownet
```

##### 3. Run the scripts containing the building recipe: 
```bash
bash ./apt_install.sh
bash ./build_environment.sh ./venv /usr/bin/flow
```
This will automatically create a simple [Python virtual environment](docs.python.org/3/library/venv.html) `./venv`

##### 4. Source the newly created virtual environment:
```bash
source ./venv/bin/activate
```

##### 5. Install the `flownet` Python module in development mode:
```bash
pip install -e .
```
Omit the `-e` flag if you want a standard installation.

> :warning: Do you want to run FlowNet through the LSF queue?
To be able to have the ERT process, that will be called by FlowNet,
run jobs via LSF correctly you will need to update your default shell's
configuration file (`.cshrc` or `.bashrc`) to automatically source your
virtual environment.

### Running FlowNet

You can run _FlowNet_ as a single command line:
```
flownet ahm ./some_config.yaml ./some_output_folder
```
Run `flownet --help` to see all possible command line argument options.

### Running webviz to check results

Before running `webviz` for the first time on your machine, you will need to to create a localhost `https` certificate by doing:
```bash
webviz certificate --auto-install --force
```

### License

FlowNet is, with a few exceptions listed below, [GPLv3](./LICENSE).

- The [Norne test data](./tests/data/norne.tar.gz) is available under the [Open Database License](http://opendatacommons.org/licenses/odbl/1.0/)
- The [FlowNet logo](./docs/_static/flownet_logo.png) is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
