  
==================
Configuration file
==================



name
====




flownet
=======

data_source
-----------

FlowNet pulls data from various sources to create a model of flow tubes. Currently the only data source implemented is an existing model, 
but this can be extended to extracting data from various continously updated databases.

simulation
~~~~~~~~~~

FlowNet will extract the data used to construct and condition the model from an existing input simulation model. 

* **input_case**: Path to the input simulation case. 
* **vectors**: Which vectors to extract from the input simulation case and use in the conditioning. The vectors available are 
  *WTHP*, *WBHP*, *WOPT*, *WGPT*, *WWPT*, *WWIT*, *WGIT*, *WOPR*, *WGPR*, *WWPR*, *WWIR*, *WGIR*. Each of the vectors added to the config 
  yaml needs a **rel_error** (relative error, defining e.g. 0.1 will yield a 10% relative error) and **min_error** (minimum error, the 
  lowest possible observation error) defined.
* **well_logs**: Boolean variable to indicate if well logs should be loaded from the input simulation model (ADD MORE) 
  (no default value, not defining it will act the same way as False would)
* **layers**: If the input simulation model is layered to the extent that there is no (or very little) communication between layers, 
  FlowNet has an option to generate separate FlowNet models for each layer. To initiate this, supply a list of lists containing the 
  start and end layer in the input simulation model for each distinct layer

Example yaml section:

.. code-block:: yaml 

  flownet:
    data_source:
      simulation:
        input_case: /path/to/simulation_model.DATA
        vectors:
          WOPR:
            rel_error: 0.1
            min_error: 50
          WGPR:
            rel_error: 0.1
            min_error: 50
        layers:
          - [1, 5]
          - [6, 10]

In this example, the input simulation model (which has been simulated with Flow or Eclipse or similar) will be found in 
*/path/to/simulation_model.DATA*, the vectors to use in the conditioning of the FlowNet model are *WOPR* and *WGPR*, each
with a relative error of 10% and minimum error of 50 (Sm3). Two FlowNet models will be created, one based on layers 1 to 5 
in the input simulation model, and one based on layers 6 to 10 in the input simulation model.

resampling
~~~~~~~~~~

Requested observation frequency, all pandas resampling options are supported, e.g. weekly (W), monthly (M), 
quarterly (Q), yearly (A). If resampling is not defined, the original data will be kept.


concave_hull
~~~~~~~~~~~~

ADD MORE HERE

constraining
------------

kriging
~~~~~~~

The permeability and porosity in the individual flow tubes in the FlowNet model can be constrained by the well logs 
using kriging (refer to the pykrige documentation for more specific documentation). The model choices here are the following:

* **enabled**: Switch to enable or disable kriging on well log data (default value is False).
* **n**: Number of kriged values in each direction. E.g, n = 10 -> 10x10x10 = 1000 values (default is 20).
* **n_lags**: Number of averaging bins for the semivariogram (default is 6).
* **anisotropy_scaling_z**: Scalar stretching value to take into account anisotropy (default is 10).
* **variogram_model**: Specifies which variogram model to use. See PyKrige documentation for valid options (default is spherical).
* **permeability_variogram_parameters**: Parameters that define the specified variogram model. Typically this will include things like 
  *sill*, *range* and *nugget*. Permeability model sill and nugget are in log scale. See PyKrige documentation for a full list of valid options. 
* **porosity_variogram_parameters**: Parameters that define the specified variogram model. See PyKrige documentation for valid options. 
  Typically this will include things like *sill*, *range* and *nugget*.


.. code-block:: yaml 

  flownet:
    constraining:
      kriging:
        enabled: false
        n: 20
        n_lags: 6
        anisotropy_scaling_z: 10
        variogram_model: spherical
        permeability_variogram_parameters:
          sill: 0.75
          range: 1000
          nugget: 0
        porosity_variogram_parameters:
          sill: 0.05
          range: 1000
          nugget: 0

phases
------

A list of phases to be present in the FlowNet model. The available phases are *oil*, *gas*, *water*, *vapoil* and *disgas*.

pvt
---

rsvd
~~~~

The path to a csv  file with RSVD input. This file can now be done either as one table used for all EQLNUM regions, 
or as one table for each EQLNUM region. The csv file needs a header with column names "depth", "rs" and "eqlnum" 
(the latter only when multiple tables are defined).

norne_static/rsvd_multiple.csv
  

cell_length
-----------

The preferred cell length of the grid cells in the flow tubes of the FlowNet model. 
To make start and end actually be the mid points of the first and last grid cell, 
the cell_length will in general only be approximately fulfilled. 
In addition, there will always be created at least two grid cells regardless of how large 
cell_length is.

  
additional_flow_nodes
---------------------

The number of additional flow nodes to add to the FlowNet network model (in addition to the well/completion nodes extracted from 
a data source. For a single FlowNet model, this should be an integer. For a layered FlowNet model, this input could either be a list
with number of items equal to the number of layers in the FlowNet model, or it could be an integer giving the total number of nodes to 
be added to the FlowNet network. In the latter case, the total number of nodes will be assigned to each layer in the FlowNet model
according to the volume inside the concave hull around the well/completion nodes in that particular layer.


: [500, 100]

additional_node_candidates
--------------------------

The number of additional nodes to create as candidates for adding one additional node (using Mitchell's best candidate algorithm). 
The Mitchell's best candidate algorithm is implemented with two options: 1) to generate *additional_node_candidates* number of candidates
every time a new node is placed, or to generate *additional_node_candidates* number of candidates first, and iteratively select the 
*additional_flow_nodes* number of candidates from this set. The latter option is faster.

  
hull_factor
-----------

The size of the FlowNet model will be highly dependent on the areal spread of the well/completion nodes in the data from the data source.
In some cases a field may only have wells placed in the centre of the field, the shallowest area. The additional nodes are placed inside the 
convex hull covered by the initial well/completion nodes. In such cases it can be of interest to increase the size of this convex hull, to 
be able to place additional nodes outside of the original convex hull. In other cases it may be of interest to make the volume to place 
additional nodes inside smaller (if you have injection wells on the rim of the field but only want addional nodes in the centre). 
The **hull_factor** will linearly scale the distance of each point from the centroid of all the points, to make a larger (or smaller) volume 
to place additional nodes in.
  
random_seed
-----------

An integer. Set this to control the numpy random number generator, to make sure that your FlowNet models are possible to regenerate 
(meaning that two FlowNet runs with the exact same input config file will produce the same FlowNet model).

perforation_handling_strategy
-----------------------------

Strategy to be used when creating perforations. Valid options are **bottom_point**, **top_point**, **multiple**, **time_avg_open_location** 
and **multiple_based_on_workovers**.

bottom_point
  Will provide the bottom point of the well (assuming it is the last open connection specified, anywhere in time).

top_point
  Will provide the top point of the well (assuming it is the first open connection specified, anywhere in time). 

multiple
  This strategy creates multiple connections per well, as many as there is data available. Connections that
  repeatedly have the same state through time are reduced to only having records for state changes.
  Be aware that this may lead to a lot of connections in the FlowNet with potentially numerical issues as a 
  result. When generating a FlowNet that is not aware of geological layering, it is questionable whether having 
  many connections per well will lead to useful results.

time_avg_open_location
  This strategy creates multiple connections per well when the well during the historic production period has been
  straddled or plugged (i.e., individual connections have been shut).

  The following steps are performed per layer:

        1. Split connections into groups of connections per well, based on their open/closing history. That is,
           connections that have seen opening or closure at the same moment in time are considered a group. This is
           done by generating a hash value based on opening state booleans through time.
        2. For each group a bounding box will be created and it will be verified that no foreign connections (i.e.,
           connections from other groups) are inside of the bounding box.
        3. If connections of other groups are found inside of the bounding box a line will be fitted through the
           connections of the group being checked and a perpendicular splitting plane will be created at the center of
           foreign connections. Two new groups now exist that both will be checked via step 2.
        4. When all groups have no foreign connections in their bounding boxes the average location of the groups 
           are returned, including their respective open/closing times.  

multiple_based_on_workovers
  This strategy bases the number of connection on historic plugs/straddles. This should allow us to model discrete steps in, 
  for example water cut, when a connection is straddled/plugged with a minimal number of connections to a FlowNet. (ADD MORE)

fast_pyscal
-----------

maybe not relevant anymore?


training_set_end_date
---------------------

The last date to be used for conditioning/training of the FlowNet network model. The date of course 
needs to be within the date range of the observations provided in the input data.

Defining this at the same time as **training_set_fraction** will raise a ValueError.


training_set_fraction
---------------------

A number between 0 and 1 defining how much of the input data should be used for conditioning/training of 
the FlowNet network model. If there are 10 years of input obervations of e.g. WOPR, a *training_set_fraction*
of 0.6 will use 6 years of the input data for training (leaving 4 years of data for validation).

Defining this at the same time as **training_set_end_date** will raise a ValueError.


fault_tolerance
---------------

The fault definitions are calculated using the following approach:

  1) Loop through all faults
  2) Perform a triangulation of all points belonging to a fault plane and store the triangles
  3) For each connection, find all triangles in its bounding box, perform ray tracing using the Möller-Trumbore intersection algorithm.
  4) If an intersection is found, identify the grid blocks that are associated with the intersection.

The **fault_tolerance** defines the minimum distance between corners of a triangle. This value 
should be set as low as possible to ensure a high resolution fault plane generation. 
However, this might lead to a very slow fault tracing process therefore one might want to increase the tolerance.
Always check that the resulting lower resolution fault plane still is what you expected.


max_distance
------------

The longest distance between two nodes to be included in the FlowNet model. Nodes that are further apart than **max_distance**
will not have a direct connection between them (default value is 1e12, i.e. very large).


max_distance_fraction
---------------------

If defined, the **max_distance_fraction** longest connections between nodes in the FlowNet model will be removed (default value is 0).

  
prod_control_mode
-----------------

Defines how the production wells are controlled in the historic production period. Available modes are *ORAT*, *GRAT*, *WRAT*, *LRAT*, *RESV*, *BHP*.
  
inj_control_mode
----------------

Defines how the injection wells are controlled in the historic period. Available modes are *RATE* and *BHP*.


angle_threshold
---------------

Angle threshold used, after Delaunay triangulation to remove sides/tubes opposite angles larger than the supplied threshold.
The idea being that for large angles, the pathway covered by the flow tube opposite a large angle will be very similar to the 
pathway covered by the two flow tubes adjacent to the large antle.

n_non_reservoir_evaluation
--------------------------

Number of points along a tube to check whether they are in non reservoir for removal purposes. ADD MORE (Something related to concave hull?)                    
                    
min_permeability
----------------

Minimum allowed permeability in mD before a tube is removed (i.e., its cells are made inactive).


hyperopt

A dictionary with parameters relater to hyper optimization of input.


n_runs
  Number of *flownet ahm* runs in one hyperopt run.

mode
  Hyperopt mode to run with. Valid options are *random*, *tpe* and *adaptive_tpe*

loss
  Dictionary with definition of the hyperopt loss function. The definitions refer to the first analysis workflow ONLY.

  - keys: List of keys, as defined in the analysis section (ert)  
  - factors: List of factors to scale the keys.
  - metric: Metric to be used in Hyperopt.
    

Example of the entire flownet part of the configuration yaml file:

.. code-block:: yaml

  flownet:
    data_source:
      simulation:
        input_case: ../input_model/norne/NORNE_ATW2013
        vectors:
          WBHP:
            rel_error: 0.05
            min_error: 10
          WOPR:
            rel_error: 0.1
            min_error: 100
          WGPR:
            rel_error: 0.1
            min_error: 100000
        well_logs: true
        layers:
          - [1, 3]
          - [4, 22]
      concave_hull: true
    constraining:
      kriging:
        enabled: false
        n: 20
        n_lags: 6
        anisotropy_scaling_z: 10
        variogram_model: spherical
        permeability_variogram_parameters:
          sill: 0.75
          range: 1000
          nugget: 0
        porosity_variogram_parameters:
          sill: 0.05
          range: 1000
          nugget: 0
    phases:
      - oil
      - gas
      - vapoil
      - disgas
      - water
    pvt:
      rsvd: norne_static/rsvd_multiple.csv
    cell_length: 100
    additional_flow_nodes: [500, 100]
    additional_node_candidates: 1000
    hull_factor: 1.2
    random_seed: 123456
    perforation_handling_strategy: multiple_based_on_workovers
    fast_pyscal: true
    training_set_end_date: 2005-01-31
    fault_tolerance: 0.0001
    max_distance_fraction: 0.10
    prod_control_mode: RESV
    inj_control_mode: RATE


ert
===






model_parameters
================


flownet:
  
  
ert:
  static_include_files: ./norne_static
  realizations:
    num_realizations: 250
    required_success_percent: 20
    max_runtime: 500
  queue:
    system: LOCAL
    max_running: 20
  ensemble_weights:
    - 4
    - 2
    - 1
    - 1
    - 1
    - 1
    - 1
    - 1
    - 1
    - 1

model_parameters:
  permeability:
    min: 1
    max: 1000
    distribution: logunif
  porosity:
    min: 0.05
    max: 0.35
  bulkvolume_mult:
    mean: 1.0
    max: 1.5
  fault_mult:
    min: 1.0e-5
    max: 1
    distribution: logunif
  relative_permeability:
    scheme: individual
    interpolate: false
    regions:
      - id: None
        swirr:
          min: 0.01
          base: 0.01
          max: 0.01
        swl:
          min: 0.05
          base: 0.05
          max: 0.05
        swcr:
          min: 0.1
          base: 0.2
          max: 0.3
        sorw:
          min: 0.2
          base: 0.25
          max: 0.3
        sorg:
          min: 0.1
          base: 0.15
          max: 0.2
        sgcr:
          min: 0.03
          base: 0.055
          max: 0.08
        krwend:
          min: 0.4
          base: 0.5
          max: 0.6
        kroend:
          min: 0.9
          base: 0.95
          max: 1.0
        krgend:
          min: 0.9
          base: 0.95
          max: 1.0
        nw:
          min: 1.5
          base: 2.25
          max: 3.0
        now:
          min: 1.5
          base: 2.25
          max: 3.0
        ng:
          min: 1.5
          base: 2.25
          max: 3.0
        nog:
          min: 1.5
          base: 2.25
          max: 3.0
  equil:
    scheme: regions_from_sim
    regions:
      - id: None
        datum_depth: 2582
        datum_pressure:
          min: 260
          max: 280
        goc_depth:
          min: 2560
          max: 2600
        owc_depth:
          min: 2670
          max: 2725
      - id: 2
        datum_depth: 2500
        datum_pressure:
          min: 250
          max: 270
        goc_depth:
          min: 2475
          max: 2525
        owc_depth:
          min: 2565
          max: 2605
      - id: 3
        datum_depth: 2582
        datum_pressure:
          min: 260
          max: 280
        goc_depth:
          min: 2560
          max: 2600
        owc_depth:
          min: 2601
          max: 2640
      - id: 4
        datum_depth: 2200
        datum_pressure:
          min: 230
          max: 250
        goc_depth:
          min: 2175
          max: 2225
        owc_depth:
          min: 2375
          max: 2425

  aquifer:
    scheme: individual
    fraction: 0.25
    delta_depth: 1000
    size_in_bulkvolumes:
      min: 1.0e-6
      max: 2
      distribution: logunif