# Change Log for FlowNet
All notable changes to `FlowNet` will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- [#221](https://github.com/equinor/flownet/pull/221) For traceability and reproducibility, the FlowNet configuration file is now automatically copied to the output folder. The content of 'pip freeze' is stored in the file 'pipfreeze.output' in the output folder.
- [#220](https://github.com/equinor/flownet/pull/220) User can now provide the historical _control mode_ in the production wells and _control mode_/_target_ in the injection wells. One value for each, applied to all wells.
- [#199](https://github.com/equinor/flownet/pull/199) Added the possibility of defining a 'base' value for all parameters related to relative permeability. This provides the opportunity to interpolate between three relative permeability models. All the 'min' values will be used to construct a pessimistic or low model, the 'base' values are used for a 'base' model, and the 'high' values are used for an optimistic or high model value. The history matching can then be done with only one (two for three phase models - oil/water and gas/oil interpolated independently) relative permeability parameter(s) per SATNUM region.
- [#197](https://github.com/equinor/flownet/pull/197) Added opening/closing of connections based on straddling/plugging/perforations through time. This PR also adds a new perforation strategy `multiple_based_on_workovers` which models the well connections with as few connections as possible taking into account opening/closing of groups of connections. 
- [#188](https://github.com/equinor/flownet/pull/188) The possibility to extract regions from an input simulation model extended to also include SATNUM regions. For relative permeability, the scheme keyword can be set to 'regions_from_sim' in the configuration file.
- [#189](https://github.com/equinor/flownet/pull/189) User can now provide both a _base_ configuration file, and an optional extra configuration file which will be used to override the base settings.

### Changed
- [#199](https://github.com/equinor/flownet/pull/199) Removed deprecated parameters in pyscal ('krowend', 'krogend') from config file. Added 'kroend' to config file. 

## [0.3.0] - 2020-09-14
### Added
- [#160](https://github.com/equinor/flownet/pull/160) Adds the possibility to extract regions from an existing model when the data source is a simulation model. For equil, the scheme key can be set to 'regions_from_sim'
- [#157](https://github.com/equinor/flownet/pull/157) Adds a new 'time-weighted average open perforation location' perforation strategy called `time_avg_open_location`. 
- [#150](https://github.com/equinor/flownet/pull/150) Adds this changelog.
- [#146](https://github.com/equinor/flownet/pull/146) Added about page to documentation with logo of industry and research institute partners.
- [#138](https://github.com/equinor/flownet/pull/138) Print message to terminal when the schedule is being generated instead of utter silence.
- [#134](https://github.com/equinor/flownet/pull/134) Egg [model](https://github.com/equinor/flownet-testdata/blob/master/egg/ci_config/assisted_history_matching.yml) added in CI.
- [#128](https://github.com/equinor/flownet/pull/128) Add possibility to use the original simulation model volume as volumetric constraints for adding new nodes as well as the option to set a maximum connection length.
- [#115](https://github.com/equinor/flownet/pull/115) Documentation of the preliminary Egg model results have been added.
- [#104](https://github.com/equinor/flownet/pull/104) Adds a new FlowNet logo.
- [#98](https://github.com/equinor/flownet/pull/98) Rock and Aquifer keywords are now allowed to be optional parameters.
- [#95](https://github.com/equinor/flownet/pull/95), [#97](https://github.com/equinor/flownet/pull/97) Relative permeability and `training_set_end_date` are now allowed to be optional. The training set definition will be checked upon parsing the configfile.
- [#90](https://github.com/equinor/flownet/pull/90) Running the analysis workflow is now optional.
- [#77](https://github.com/equinor/flownet/pull/77) Adds a warning message when the `WSTAT` keyword is not present in the original simulation data deck. The default status is `OPEN`.

### Fixed
- [#141](https://github.com/equinor/flownet/pull/141) Fixed misplacement of `WCONINJH` target (`RATE` / `BHP`). The default `RATE` is however still always used.
- [#137](https://github.com/equinor/flownet/pull/137) After having moved all example data to [flownet-testdata](https://github.com/equinor/flownet-testdata) the examples folder had no use longer. Removed.
- [#136](https://github.com/equinor/flownet/pull/136) The ERT queue name was wrongly required in the config file. It is now allowed to have no value.
- [#119](https://github.com/equinor/flownet/pull/119) Workaround for [equinor/libres#984](https://github.com/equinor/libres/issues/984)
- [#92](https://github.com/equinor/flownet/pull/92) Fixed problem where the last production well would not correctly be shut.
- [#88](https://github.com/equinor/flownet/pull/88), [#106](https://github.com/equinor/flownet/pull/92), [#108](https://github.com/equinor/flownet/pull/108), [#110](https://github.com/equinor/flownet/pull/110) Fixed bugs in parsing the config that were a result of a changed behavior in configsuite where an item that has subitems with subitems, no longer can be tested for all `None` as the subitem with subitems in fact is a `NamedDict` with all `None`.
- [#89](https://github.com/equinor/flownet/pull/89), [#91](https://github.com/equinor/flownet/pull/91), [#99](https://github.com/equinor/flownet/pull/99) Updates and fixes in git workflow.
- [#81](https://github.com/equinor/flownet/pull/81) Various bugfixes in the analysis workflow are now solved.
- [#75](https://github.com/equinor/flownet/pull/75) Resampling via the config path is no longer supported. This PR removes also the resampling option from the config file parser.
 
### Changed
- [#147](https://github.com/equinor/flownet/pull/147) The observation uncertainty now only needs to be specified for the vectors that are actually used in the conditioning. This PR also adds possibility of defining uncertainty for WWIR and WGIR into the flownet config YAML.
- [#132](https://github.com/equinor/flownet/pull/132) The observation uncertainty, used for conditioning models, is now exposed to the user via the configuration file instead of hidden in the `observations.yamlobs.jinja2` template. This PR also introduced a wider usage of passing around the `ConfigSuit.snapshot`.
- [#128](https://github.com/equinor/flownet/pull/117) Simulation models and configuration files used in CI are now coming from [flownet-testdata](https://github.com/equinor/flownet-testdata) and are thus no longer an integral part of the flownet repository.
- [#125](https://github.com/equinor/flownet/pull/125) AHM and prediction `subprocess` call are now being performed through a common `run_ert_subprocess` function call.
- [#123](https://github.com/equinor/flownet/pull/123) Significant speed-up of the fault raytracing by limiting the ray tracing to the bounding box of individual connections instead of ray tracing over all triangles of all faults for each connection.
- [#111](https://github.com/equinor/flownet/pull/111) Phases present in the FlowNet model are to be set manually in configuration by the user.
- [#63](https://github.com/equinor/flownet/pull/63) The default simulator name has now been changed to _Flow_.
- [#80](https://github.com/equinor/flownet/pull/80) FlowNet now supports relative paths in the config file.

## [0.2.0] - 2020-06-25

- No changelog was maintained until this release
