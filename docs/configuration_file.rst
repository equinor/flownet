  
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
--------

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


runpath
-------

(the default runpath  is *output/runpath/realization-%d/iter-%d*)

enspath
-------

(the default enspath is *output/storage*)

eclbase
-------

(the default eclbase is *./eclipse/model/FLOWNET_REALIZATION*)

static_include_files
--------------------

(the default is pathlib.Path(os.path.dirname(os.path.realpath(__file__)))/"static_include_files"/".."/ "static")


realizations
------------

A dictionary with some key/value pairs that control the number of realizations to submit to ERT, and how these 
should be treated as successes/failures.

num_realizations
~~~~~~~~~~~~~~~~

Number of realizations to start with in the first iteration

required_success_percent
~~~~~~~~~~~~~~~~~~~~~~~~

The percentage of completed realizations needed for an iteration to be deemed as successful. After a successful
iteration, the algorithm will moved on to the next iteration (the default value is 20).


max_runtime
~~~~~~~~~~~

The number of seconds allowed for a single realization. After the given amount of seconds, the realization in
question will be deemed as unsuccessful (the default value is 300). This is to avoid having to wait a long time for realizations with numerical problems.

queue
-----

Information about where to perform the reservoir simulations. Currently there are two possibilities, namely local or lsf.

system
~~~~~~

Controls where the reservoir simulation jobs are executed. The keyword can take the values *lsf* or *local*. The lsf option
will submit jobs to the lsf cluster at your location. This keyword has no default value, and needs to be defined.

server
~~~~~~

The server the reservoir simulation jobs will be sent to. The jobs will be sent using shell commands (*bsub/bjobs/bkill*).


name
~~~~

The name of the simulation queue on the server where the reservoir simulation jobs will be sent.


max_running
~~~~~~~~~~~

The maximum number of simulation jobs executed simulataneously.


ensemble_weights
----------------

A list with weights assigned to the iteration in the ES MDA algorithm.

yamlobs
-------

Name of the observations file used by fmu ensemble and webviz (default value *./observations.yamlobs*).

analysis
--------

A list of analysis workflows to run, to assess the quality of the history matching.

metric
~~~~~~

List of accuracy metrics to be computed in FlowNet analysis workflow. Supported metrics: MSE, RMSE, NRMSE, MAE, NMAE, R2.


quantity
~~~~~~~~

List of summary vectors for which accuracy is to be computed.

start
~~~~~

Start date in YYYY-MM-DD format.

end
~~~

End date in YYYY-MM-DD format.

outfile
~~~~~~~

The filename of the output of the workflow. In case multiple analysis workflows are run this name should be unique.


model_parameters
================

The different parameters to be tuned are defined in the **model_parameters** 
section of the FlowNet config yaml. At present, the model can be parameterized 
with the following required parameters:

* Permeability
* Porosity
* Bulk volume multipliers
* Saturation endpoints, relative permeability endpoints and Corey exponents
* Datum pressures and contacts

For permeability, porosity and bulk volume multipliers there is also an option to
include a regional (based on an existing grid parameter) or global multiplier as well.

In addition there are a few optional parameters that may be included:

* Fault multipliers
* Aquifer size (relative to the bulk volume in the model)
* Rock compressibility

All parameters need an initial guess on what values they can take. This is referred to as the prior 
probability distribution.

.. _prior:

The following keys are available for defining the different prior distributions: 

distribution
  The type of probability distribution. 

min
  The minimum value of the chosen prior probability distribution. 

max
  The maximum value of the chosen prior probability distribution. 

base
  The mode of the prior probability distribution
  
mean
  The mean or expected value of the prior probability distribution

stddev
  The standard deviation of the prior probability distributions

Their usage will be the same for all the model parameters, except for when using 
the interpolation option for relative permeability. In that case min, base, and max will 
have a different meaning, which will be described in more detail later. There is also an 
additional keyword *low_optimistic* which only is meaningful to define when using the 
interpolation option for relative permeability.

The table below describes the available prior probability distributions, and how they
should be defined in the FlowNet config yaml. If one choice of probability distribution
has several rows in the table, it means that there are more than one way to define that 
specific probability distribution. The **uniform** distribution can for example be defined
by providing the *min* and *max* values, but it can also be defined by providing the *min* 
and *mean* values (where FlowNet will calculate the *max* value), or by providing the
*mean* and *max* values.

+---------------------------+------------------+------+------+------+------+------+
| Probability distributions | distribution     | min  | max  | mean | base |stddev|
+===========================+==================+======+======+======+======+======+
| Normal                    | normal           |      |      |   x  |      |   x  |        
+---------------------------+------------------+------+------+------+------+------+
| Truncated normal          | truncated_normal |  x   |  x   |   x  |      |   x  |        
+---------------------------+------------------+------+------+------+------+------+
| Uniform                   | uniform          |  x   |  x   |      |      |      |        
+                           +                  +------+------+------+------+------+
|                           |                  |  x   |      |   x  |      |      |        
+                           +                  +------+------+------+------+------+
|                           |                  |      |  x   |   x  |      |      |        
+---------------------------+------------------+------+------+------+------+------+
| Log-uniform               | logunif          |  x   |  x   |      |      |      |       
+                           +                  +------+------+------+------+------+
|                           |                  |  x   |      |   x  |      |      |        
+                           +                  +------+------+------+------+------+
|                           |                  |      |  x   |   x  |      |      |        
+---------------------------+------------------+------+------+------+------+------+
| Triangular                | triangular       |  x   |  x   |      |  x   |      |        
+                           +                  +------+------+------+------+------+
|                           |                  |  x   |  x   |   x  |      |      |        
+                           +                  +------+------+------+------+------+
|                           |                  |      |  x   |   x  |  x   |      |        
+                           +                  +------+------+------+------+------+
|                           |                  |  x   |      |   x  |  x   |      |        
+---------------------------+------------------+------+------+------+------+------+
| Log-normal                | lognormal        |      |      |   x  |      |  x   |        
+---------------------------+------------------+------+------+------+------+------+
| Constant (Dirac)          | const            |      |      |      |   x  |      |        
+---------------------------+------------------+------+------+------+------+------+


permeability
------------

Defines the prior probability distribution for permeability as described in `prior`_. Only one distribution
should be defined, and it will be used for all flow tubes. The permeability values for
different flow tubes are drawn independently.

permeability_regional_scheme
----------------------------

This keyword can take the values *individual*, *global* and *regions_from sim*. The default value is *individual*, meaning that
no regional permeability multipliers will be applied. Setting the value to global means that there will be one global permeability 
multiplier on top of the individual ones. The last option, *regions_from_sim*, gives the possibility of introducing regional
permeability multipliers following the region definitions in a grid parameter inside an existing simulation model. When using 
*regions_from_sim*, the name of the grid parameter should be given in the *permeability_parameter_from_sim_model* keyword.
The prior distribution for the regional permeability multiplier needs to be defined with the *permeability_regional* keyword.


permeability_regional
---------------------

Defines a prior probability distribution (as described in `prior`_) for a regional permeability multiplier. Only one distribution
should be defined, and it will be used for all regions defined. 


permeability_parameter_from_sim_model
----------------------------------------

The name of the grid parameter in an existing reservoir simulation model to extract regions from to generate regional permeability multipliers.



+------------------------------------------------------+----------------------------------+------------------------------------------------------+
| Available options in config yaml                     | Example of usage                 | Example of usage                                     |
+------------------------------------------------------+----------------------------------+------------------------------------------------------+
| .. code-block:: yaml                                 | .. code-block:: yaml             | .. code-block:: yaml                                 |
|                                                      |                                  |                                                      |
|    flownet:                                          |    flownet:                      |    flownet:                                          |
|      model_parameters:                               |      model_parameters:           |      model_parameters:                               |
|        permeability:                                 |        permeability:             |        permeability:                                 |
|          min:                                        |          min: 10                 |          min: 10                                     |
|          max:                                        |          max: 1000               |          mean: 100                                   |
|          base:                                       |          distribution: logunif   |          distribution: uniform                       |
|          mean:                                       |                                  |        permeability_regional_scheme: regions_from_sim|
|          stddev:                                     |                                  |        permeability_regional:                        |
|          distribution:                               |                                  |          min: 0.5                                    |
|        permeability_regional_scheme:                 |                                  |          max: 1.5                                    |
|        permeability_regional:                        |                                  |        permeability_parameter_from_sim_model: FIPNUM |
|          min:                                        |                                  |                                                      |
|          max:                                        |                                  |                                                      |
|          base:                                       |                                  |                                                      |
|          mean:                                       |                                  |                                                      |
|          stddev:                                     |                                  |                                                      |
|          distribution:                               |                                  |                                                      |
|        permeability_parameter_from_sim_model:        |                                  |                                                      |
+------------------------------------------------------+----------------------------------+------------------------------------------------------+


porosity
--------
Defines the prior probability distribution for porosity. Only one distribution
should be defined, and it will be used for all flow tubes. The porosity values for
different flow tubes are drawn independently.


porosity_regional_scheme
------------------------


porosity_regional
-----------------


porosity_parameter_from_sim_model
---------------------------------



+------------------------------------------------------+----------------------------------+--------------------------------------------------------+
| Available options in config yaml                     | Example of usage                 | Example of usage                                       |
+------------------------------------------------------+----------------------------------+--------------------------------------------------------+
| .. code-block:: yaml                                 | .. code-block:: yaml             | .. code-block:: yaml                                   |
|                                                      |                                  |                                                        |
|    flownet:                                          |    flownet:                      |    flownet:                                            |
|      model_parameters:                               |      model_parameters:           |      model_parameters:                                 |
|        porosity:                                     |        porosity:                 |        porosity:                                       |
|          min:                                        |          min: 0.15               |          min: 0.20                                     |
|          max:                                        |          max: 0.35               |          max: 0.40                                     |
|          base:                                       |          distribution: uniform   |          distribution: uniform                         |
|          mean:                                       |                                  |        porosity_regional_scheme: regions_from_sim      |
|          stddev:                                     |                                  |        porosity_regional:                              |                                  | 
|          distribution:                               |                                  |          min: 0.5                                      |                                  |
|        porosity_regional_scheme:                     |                                  |          mean: 1                                       |
|        porosity_regional:                            |                                  |          max: 2                                        |
|          min:                                        |                                  |          distribution: triangluar                      |
|          max:                                        |                                  |        porosity_parameter_from_sim_model: FIPNUM       |
|          base:                                       |                                  |                                                        |
|          mean:                                       |                                  |                                                        |
|          stddev:                                     |                                  |                                                        |
|          distribution:                               |                                  |                                                        |
|        porosity_parameter_from_sim_model:            |                                  |                                                        |
+------------------------------------------------------+----------------------------------+----------------------------------+---------------------+



bulkvolume_mult
---------------

FlowNet has two options in the config yaml deciding how the bulk volume should be
distributed initially. These options are:

* **tube_length**: Here the bulk volume covered by the convex hull of the FlowNet will be divided equally to all active cells
* **voronoi_per_tube**: This is based on an input simulation model. The bulk volume of each cell in the input simulation model 
  will be assigned to the nearest cell in any flow tube in the FlowNet model. When all the bulk volume in the input simulation 
  model have been assigned to cells in the FlowNet model, the total bulk volume assigned to each flow tube in the FlowNet model 
  is distributed evenly to all cells in that flow tube.

Each flow tube can be thought to represent the bulk volume in the region between the two nodes it connects. 
There could be several reasons why the bulk volume in a flow tube should be adjusted up or down, hence there 
is a need to be able to tune the bulk volume for efficient history matching.

This multiplier will act on top of that initial distribution of 
bulk volume.

This part of the config file defines the prior probability distribution 
for a bulk volume multiplier. Only one distribution
should be defined, and it will be used for all flow tubes. The values for
different flow tubes are drawn independently.


bulkvolume_mult_regional_scheme
-------------------------------


bulkvolume_mult_regional
------------------------


bulkvolume_mult_parameter_from_sim_model
----------------------------------------



+------------------------------------------------------+----------------------------------+----------------------------------------------------------+
| Available options in config yaml                     | Example of usage                 | Example of usage                                         |
+------------------------------------------------------+----------------------------------+----------------------------------------------------------+
| .. code-block:: yaml                                 | .. code-block:: yaml             | .. code-block:: yaml                                     |
|                                                      |                                  |                                                          |
|    flownet:                                          |    flownet:                      |    flownet:                                              |
|      model_parameters:                               |      model_parameters:           |      model_parameters:                                   |
|        bulkvolume_mult:                              |        bulkvolume_mult:          |        bulkvolume_mult:                                  |
|          min:                                        |          min: 0.2                |          mean: 1                                         |
|          max:                                        |          max: 4                  |          stddev: 0.1                                     |
|          base:                                       |          distribution: uniform   |          min: 0.2                                        |
|          mean:                                       |                                  |          max: 2                                          |
|          stddev:                                     |                                  |          distribution: truncated_normal                  |
|          distribution:                               |                                  |        bulkvolume_mult_regional_scheme: regions_from_sim |
|        bulkvolume_mult_regional_scheme:              |                                  |          mean: 1                                         |
|        bulkvolume_mult_regional:                     |                                  |          max: 2                                          |
|          min:                                        |                                  |          distribution: triangluar                        |
|          max:                                        |                                  |        bulkvolume_mult_parameter_from_sim_model: FIPNUM  |
|          base:                                       |                                  |                                                          |
|          mean:                                       |                                  |                                                          |
|          stddev:                                     |                                  |                                                          |
|          distribution:                               |                                  |                                                          |
|        bulkvolume_mult_parameter_from_sim_model:     |                                  |                                                          |
+------------------------------------------------------+----------------------------------+----------------------------------------------------------+
        

relative_permeability
---------------------

FlowNet currently uses Corey correlations for generating relative permeability input curves for Flow. At a later 
stage LET parametrization may also be implemented.


scheme
~~~~~~

The scheme parameter decides how many sets of relative permeability curves to generate as
input to Flow. There are three options. With **scheme: global** only one set of relative 
permeability curves will be generated, and applied to all flow tubes in the model. With
**scheme: individual** all flow tubes in the model will have its own set of relative permeability
curves. With **scheme: regions_from_sim** FlowNet will extract the SATNUM regions from the 
input model provided, and assign the same set of relative permeability curves to all flow tubes 
that are (mostly) located within the same SATNUM region. The default value is global.

interpolate
~~~~~~~~~~~

SCAL experts will often provide three sets of relative permeability curves (one pessimistic set, 
one base set and one optimistic set) to run sensitivities on a reservoir model. 
This introduces the option of generating new sets of relative permeability curves within the 
envelope created by the low/base/high sets of curves by using an interpolation parameter 
(potentially two interpolation parameters in three phase models). This will limit the number of 
history matching parameters, especially when the number of SATNUM regions is large. The default 
value is False. A parameter value on the interval [-1,0) will interpolate all input parameters 
(Corey exponents, saturation endpoints and relative permeability endpoints) linearly between the 
value in the low model and the base model. A parameter value on the interval [0,1] will interpolate
between the base model and the high model. 

independent_interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~

If **interpolate** is set to **True** and the model has three active phases, this parameter will
decide whether or not the interpolation for water/oil relative permeability and gas/oil relative 
permeability will be performed independently. The default value is False.


region_parameter_from_sim_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The name of the regions grid parameter in the simulation model to base the relative permeability 
region parameter in the FlowNet model on (the default parameter is SATNUM).

swcr_add_to_swl
~~~~~~~~~~~~~~~

Allows for calculating SWCR by adding a number to SWL. Especially useful to avoid non-physical values 
when defining prior distributions. If this parameter is set to true, the numbers defined under swcr 
will be used to define a prior distribution for the delta value added to SWL, instead of defining the 
prior distribution for SWCR directly (default value is False).

krwmax_add_to_krwend
~~~~~~~~~~~~~~~~~~~~

Allows for calculating KRWMAX by adding a number to KRWEND. Especially useful to avoid non-physical 
values when defining prior distributions. If this parameter is set to true, the numbers defined 
under KRWMAX will be used to define a prior distribution for the delta value added to KRWEND, 
instead of defining the prior distribution for KRWMAX directly (the default value is False).
  
regions
~~~~~~~
  
This is a list where each list element will contain information about the saturation endpoints 
and relative permeability endpoints within one SATNUM region, in addition to a region identifier. The 
endpoints are shown in two figures below for clarification.
The number of list elements needs to be equal to the number of SATNUM regions in the model,
unless one of the regions is defined with identifier *None*. 

* id: Region identifier. Default value is None.
* swirr: The irreducible water saturation. 
* swl: Connate water saturation. 
* swcr: Critical water saturation. 
* sorw: Residual oil saturation (that cannot be displaced by water). 
* krwend: Maximum relative permeability for water. 
* kroend: Maximum relative permeability for oil. 
* nw, now, ng, nog: Exponents in Corey parametrization. 
* sorg: Residual oil saturation (that cannot be displaced by gas). 
* sgcr: Critical gas saturation. 
* krgend: Maximum relative permeability for gas
  
A water/oil model needs *swirr*, *swl*, *swcr*, *sorw*, *nw*, *now*, *krwend* and *kroend* to be defined.
An oil/gas model needs *swirr*, *swl*, *sgcr*, *sorg*, *ng*, *nog*, *krgend* and *kroend* to be defined.
A three phase model needs all 13 relative permeability parameters to be defined.

All of the relative permeability parameters above should have prior distributions defined according to `prior`_.


  
.. figure:: https://equinor.github.io/pyscal/_images/gasoil-endpoints.png
  
   Visualization of the gas/oil saturation endpoints and gas/oil relative permeability endpoints as modelled by pyscal. 

.. figure:: https://equinor.github.io/pyscal/_images/wateroil-endpoints.png
  
   Visualization of the water/oil saturation endpoints and water/oil relative permeability endpoints as modelled by pyscal. 



When using the interpolation option for relative permeability, some of the keywords related to choice 
of prior distribution above have a different meaning. This applies to **min**, **base**, and **max**. 
There is also an additional keyword **low_optimistic** which only is meaningful to define for relative permeability.

Each of the input parameters needs a low, base, and high value to be defined. This is done through
the **min** (low), **base** and **max** (high) keywords. 
For some parameters a low numerical value is favorable. This can be indicated by setting 
**low_optimistic** to **True** for that parameter (the default value of low_optimistic is False).



+----------------------------------------+----------------------------------+----------------------------------+
| Available options in config yaml       | Example of usage                 | Example of usage                 |
+----------------------------------------+----------------------------------+----------------------------------+
|                                        |                                  |                                  |
| .. code-block:: yaml                   | .. code-block:: yaml             | .. code-block:: yaml             |
|                                        |                                  |                                  |
|  flownet:                              |  flownet:                        |  flownet:                        |
|    model_parameters:                   |    model_parameters:             |    model_parameters:             |
|      relative_permeability:            |      relative_permeability:      |      relative_permeability:      |
|        scheme:                         |        scheme: global            |        scheme: regions_from_sim  |
|        region_parameter_from_sim_model:|        regions:                  |        interpolate: true         |
|        swcr_add_to_swl:                |          id: None                |        regions:                  |        
|        krwmax_add_to_krwend:           |          swirr:                  |          id: None                |
|        interpolate:                    |            min:  0.01            |          swirr:                  |
|        independent_interpolation:      |            max:  0.03            |            min:  0.01            |
|        regions:                        |          swl:                    |            base: 0.02            |
|          id:                           |            min:  0.03            |            max:  0.03            |
|          swirr:                        |            max:  0.05            |          swl:                    |
|            min:                        |          swcr:                   |            min:  0.03            |
|            max:                        |            min:  0.09            |            base: 0.04            |
|            mean:                       |            max:  0.15            |            max:  0.05            |
|            base:                       |          sorw:                   |          swcr:                   |
|            stddev:                     |            min:  0.2             |            min:  0.09            |
|            distribution:               |            max:  0.3             |            base: 0.12            |
|            low_optimistic:             |          nw:                     |            max:  0.15            |
|          swl:                          |            min:  1.5             |          sorw:                   |
|            <same as for swirr>         |            max:  3.0             |            min:  0.2             |
|          swcr:                         |          now:                    |            base: 0.25            |
|            <same as for swirr>         |            min:  1.5             |            max:  0.3             |
|          sorw:                         |            max:  3.0             |          nw:                     |
|            <same as for swirr>         |          krwend:                 |            min:  1.5             |
|          krwend:                       |            min:  0.4             |            base: 2.25            |
|            <same as for swirr>         |            max:  0.6             |            max:  3.0             |
|          kroend:                       |          kroend:                 |          now:                    |
|            <same as for swirr>         |            min:  0.9             |            min:  1.5             |
|          no:                           |            max:  1.0             |            base: 2.25            |
|            <same as for swirr>         |                                  |            max:  3.0             |
|          now:                          |                                  |          krwend:                 |
|            <same as for swirr>         |                                  |            min:  0.4             |
|          sorg:                         |                                  |            base: 0.5             |
|            <same as for swirr>         |                                  |            max:  0.6             |
|          sgcr:                         |                                  |          kroend:                 |
|            <same as for swirr>         |                                  |            min:  0.9             |
|          ng:                           |                                  |            base: 0.95            |
|            <same as for swirr>         |                                  |            max:  1.0             |
|          nog:                          |                                  |                                  |
|            <same as for swirr>         |                                  |                                  |
|          krgend:                       |                                  |                                  |
|            <same as for swirr>         |                                  |                                  |
+----------------------------------------+----------------------------------+----------------------------------+
									     
									     
									     
Equilibration
-------------

This keyword contains information regarding the equilibration regions in the FlowNet model.

scheme
~~~~~~

The scheme parameter decides how many equilibration regions to generate as
input to Flow. There are three options. With **scheme: global** the model will only have one  
equilibration region, and applied to all flow tubes in the model. With
**scheme: individual** all flow tubes in the model will act as its own equilibration region. 
With **scheme: regions_from_sim** FlowNet will extract the EQLNUM regions from the 
input model provided, and assign equilibraion regions to all flow tubes accordingly. 
The default value is global.


region_parameter_from_sim_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The name of the regions grid parameter in the simulation model to base the equilibration
region parameter in the FlowNet model on (the default parameter is SATNUM).


regions
~~~~~~~

This is a list where each list element will contain information about the datum depth, datum pressure and 
fluid contacts within one equilibration region, in addition to a region identifier.
The number of list elements needs to be equal to the number of EQLNUM regions in the model,
unless one of the regions is defined with identifier *None*. 
 
* id: Region identifier. Default value is None.
* datum_depth: Datum or reference depth in the equilibrium region.
* datum_pressure: Datum or reference pressure in the equilibrium region.
* owc_depth: Depth of the oil/water contact in the equilibrium region.
* goc_depth: Depth of the gas/oil contact in the equilibrium region.
* gwc_depth: Depth of the gas/water contact in the equilibrium region.

The *datum depth* is just a number. The *datum pressure* and the different contacts 
should be entered with a prior probability distribution according to the information in `prior`_.


+-----------------------------------------+----------------------------------+
| Available options in config yaml        | Example of usage                 |
+-----------------------------------------+----------------------------------+
|                                         |                                  |
| .. code-block:: yaml                    | .. code-block:: yaml             |
|                                         |                                  |
|  flownet:                               |  flownet:                        |
|    model_parameters:                    |    model_parameters:             |
|      equil:                             |      equil:                      |
|        scheme:                          |        scheme: global            |
|        region_parameter_from_sim_model: |          regions:                |
|        regions:                         |            id: None              |
|          id:                            |            datum_depth: 2500     |
|          datum_depth:                   |            datum_pressure:       |
|          datum_pressure:                |              min: 250            |
|            min:                         |              max: 270            |
|            max:                         |            owc_depth:            |
|            mean:                        |              min: 2565           |
|            base:                        |              max: 2605           |
|            stddev:                      |            goc_depth:            |
|            distribution:                |              min: 2475           |
|          owc_depth:                     |              max: 2525           |
|            min:                         |            id: 1                 |
|            max:                         |            datum_depth: 2582     |
|            mean:                        |            datum_pressure:       |
|            base:                        |              min: 260            |
|            stddev:                      |              max: 280            |
|            distribution:                |            owc_depth:            |
|          goc_depth:                     |              min: 2670           |
|            same as for owc_depth        |              max: 2725           |
|          gwc_depth:                     |            goc_depth:            |
|            same as for owc_depth        |              min: 2560           |
|                                         |              max: 2600           |
|                                         |                                  |
+-----------------------------------------+----------------------------------+


Fault multiplier
----------------
Defines the prior probability distribution for fault transmissibility multipliers. Only one distribution
should be defined, and it will be used for all faults in the model. The fault transmissibilities for different
faults are drawn independently.

+----------------------------------+----------------------------------+----------------------------------------+
| Available options in config yaml | Example of usage                 | Example of usage                       |
+----------------------------------+----------------------------------+----------------------------------------+
| .. code-block:: yaml             | .. code-block:: yaml             | .. code-block:: yaml                   |
|                                  |                                  |                                        |
|    flownet:                      |    flownet:                      |    flownet:                            |
|      model_parameters:           |      model_parameters:           |      model_parameters:                 |
|        fault_mult:               |        fault_mult:               |        fault_mult:                     |
|          min:                    |          min: 0.0001             |          min: 0                        |
|          max:                    |          max: 1                  |          max: 1                        |
|          base:                   |          distribution: logunif   |          base: 0.1                     | 
|          mean:                   |                                  |          distribution: triangular      |
|          stddev:                 |                                  |                                        |
|          distribution:           |                                  |                                        |
+----------------------------------+----------------------------------+----------------------------------------+


Rock compressibility
--------------------

Rock compressibility can be included by defining the *reference pressure* and the 
minimum and maximum value. The minimum and maximum value will be used to define
a uniform distribution, from which all realizations of the FlowNet will be assigned 
a value.

+----------------------------------+----------------------------------+
| Available options in config yaml | Example of usage                 |
+----------------------------------+----------------------------------+
|                                  |                                  |
| .. code-block:: yaml             | .. code-block:: yaml             |
|                                  |                                  |
|  flownet:                        |  flownet:                        |
|    model_parameters:             |    model_parameters:             |
|      rock_compressibility:       |      rock_compressibility:       |
|        reference_pressure:       |        reference_pressure:       |
|        min:                      |        min:                      |
|        max:                      |        max:                      |
|                                  |                                  |
+----------------------------------+----------------------------------+


Aquifer
-------

+----------------------------------+----------------------------------+
| Available options in config yaml | Example of usage                 |
+----------------------------------+----------------------------------+
|                                  |                                  |
| .. code-block:: yaml             | .. code-block:: yaml             |
|                                  |                                  |
|  flownet:                        |  flownet:                        |
|    model_parameters:             |    model_parameters:             |
|      aquifer:                    |      aquifer:                    |
|        scheme:                   |        scheme: individual        |
|        fraction:                 |        fraction: 0.25            |
|        delta_depth:              |        delta_depth: 1000         |
|        size_in_bulkvolumes:      |        size_in_bulkvolumes:      |
|           min:                   |          min: 1.0e-4             |
|           max:                   |          max: 2                  |
|           mean:                  |                                  |
|           base:                  |                                  |
|           stddev:                |                                  |
|           distribution:          |                                  |
|                                  |                                  |
+----------------------------------+----------------------------------+

scheme:
  The **scheme** parameter decides the number of aquifers. Setting scheme 
  to *individual* means that all aquifer connections goes to individual aquifers.
  Setting scheme to *global* means that all aquifer connections goes to one single 
  aquifer.

fraction:
  Decides how many nodes the aquifer(s) should connect to. Currently the implementation
  relies on depth only, selecting the *fraction* deepest nodes in the FlowNet.

delta_depth:
  Decides the depth of the aquifer node(s). When using the global option, a single aquifer node
  will be placed *delta_depth* below the average position of all the nodes it should connect to.
  When using the individual option, one aquifer node will be placed *delta_depth* below each of
  the selected FlowNet nodes.

size_in_bulkvolumes:
  The size of the aquifer, relative to the bulk volume of the FlowNet the aquifer nodes connect to.
  



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
