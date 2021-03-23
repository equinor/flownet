==================
Configuration file
==================

The configuration files follows the `YAML standard <https://yaml.org/>`_.

.. configsuite::
    :module: flownet.config_parser._config_parser.create_schema_without_arguments


Model parameters - defining prior distributions in FlowNet
==========================================================

The different parameters to be tuned are defined in the **model_parameters** 
section of the FlowNet config yaml. At present, the model can be parameterized 
with the following required parameters:

* Permeability
* Porosity
* Bulk volume multipliers
* Fault multipliers
* Saturation endpoints, relative permeability endpoints and Corey exponents
* Datum pressures and contacts

In addition there are a few optional parameters that may be included:
* Rock compressibility
* Aquifer size (relative to the bulk volume in the model)


All parameters need an initial guess 
on what values they can take. This is referred to as the prior probability distribution.

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

Their usage will be the same for all the required model parameters, except for when using 
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



Permeability
------------

Defines the prior probability distribution for permeability. Only one distribution
should be defined, and it will be used for all flow tubes. The permeability values for
different flow tubes are drawn independently.


+----------------------------------+----------------------------------+----------------------------------+----------------------------------+
| Available options in config yaml | Example of usage                 | Example of usage                 | Example of usage                 |
+----------------------------------+----------------------------------+----------------------------------+----------------------------------+
| .. code-block:: yaml             | .. code-block:: yaml             | .. code-block:: yaml             | .. code-block:: yaml             |
|                                  |                                  |                                  |                                  |
|    flownet:                      |    flownet:                      |    flownet:                      |    flownet:                      |
|      model_parameters:           |      model_parameters:           |      model_parameters:           |      model_parameters:           |
|        permeability:             |        permeability:             |        permeability:             |        permeability:             |
|          min:                    |          min: 10                 |          min: 10                 |          min: 10                 |
|          max:                    |          max: 1000               |          mean: 100               |          base: 50                |
|          base:                   |          distribution: logunif   |          distribution: uniform   |          max: 200                |
|          mean:                   |                                  |                                  |          distribution: triangular|
|          stddev:                 |                                  |                                  |                                  | 
|          distribution:           |                                  |                                  |                                  |
+----------------------------------+----------------------------------+----------------------------------+----------------------------------+


Porosity
--------
Defines the prior probability distribution for porosity. Only one distribution
should be defined, and it will be used for all flow tubes. The porosity values for
different flow tubes are drawn independently.

+----------------------------------+----------------------------------+----------------------------------+----------------------------------+
| Available options in config yaml | Example of usage                 | Example of usage                 | Example of usage                 |
+----------------------------------+----------------------------------+----------------------------------+----------------------------------+
| .. code-block:: yaml             | .. code-block:: yaml             | .. code-block:: yaml             | .. code-block:: yaml             |
|                                  |                                  |                                  |                                  |
|    flownet:                      |    flownet:                      |    flownet:                      |    flownet:                      |
|      model_parameters:           |      model_parameters:           |      model_parameters:           |      model_parameters:           |
|        porosity:                 |        porosity:                 |        porosity:                 |        porosity:                 |
|          min:                    |          min: 0.15               |          mean: 0.25              |          min: 0.15               |
|          max:                    |          max: 0.35               |          stddev: 0.03            |          mean: 0.22              |
|          base:                   |          distribution: uniform   |          distribution: normal    |          max: 0.31               |
|          mean:                   |                                  |                                  |          distribution: triangular|
|          stddev:                 |                                  |                                  |                                  | 
|          distribution:           |                                  |                                  |                                  |
+----------------------------------+----------------------------------+----------------------------------+----------------------------------+



Bulk volume multiplier
----------------------

Each flow tube can be thought to represent the bulk volume in the region between the 
two nodes it connects. There could be several reasons why the bulk volume in a flow tube 
should be adjusted up or down, hence there is a need to be able to tune the bulk volume
for efficient history matching.

FlowNet also has an options in the config yaml deciding how the bulk volume should be
distributed initially. This multiplier will act on top of that initial distribution of 
bulk volume.

This part of the config file defines the prior probability distribution 
for a bulk volume multiplier. Only one distribution
should be defined, and it will be used for all flow tubes. The values for
different flow tubes are drawn independently.

+----------------------------------+----------------------------------+----------------------------------------+
| Available options in config yaml | Example of usage                 | Example of usage                       |
+----------------------------------+----------------------------------+----------------------------------------+
| .. code-block:: yaml             | .. code-block:: yaml             | .. code-block:: yaml                   |
|                                  |                                  |                                        |
|    flownet:                      |    flownet:                      |    flownet:                            |
|      model_parameters:           |      model_parameters:           |      model_parameters:                 |
|        bulkvolume_mult:          |        bulkvolume_mult:          |        bulkvolume_mult:                |
|          min:                    |          min: 0.2                |          mean: 1                       |
|          max:                    |          max: 4                  |          stddev: 0.1                   |
|          base:                   |          distribution: uniform   |          min: 0.2                      |
|          mean:                   |                                  |          max: 2                        |
|          stddev:                 |                                  |          distribution: truncated_normal|
|          distribution:           |                                  |                                        |
+----------------------------------+----------------------------------+----------------------------------------+

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

        

Saturation endpoints, relative permeability endpoints and Corey exponents
-------------------------------------------------------------------------

FlowNet currently uses Corey correlations for generating relative permeability input curves for Flow. At a later 
stage LET parametrization may also be implemented.

The input related to relative permeability modelling has its own section in the config yaml file. 

+----------------------------------+----------------------------------+
| Available options in config yaml | Example of usage                 |
+----------------------------------+----------------------------------+
|                                  |                                  |
| .. code-block:: yaml             | .. code-block:: yaml             |
|                                  |                                  |
|  flownet:                        |  flownet:                        |
|    model_parameters:             |    model_parameters:             |
|      relative_permeability:      |      relative_permeability:      |
|        scheme:                   |        scheme: global            |
|        interpolate:              |        interpolate: true         |
|        independent_interpolation:|        regions:                  |
|        regions:                  |          swirr:                  |
|          id:                     |            min:  0.01            |
|          swirr:                  |            max:  0.03            |
|            min:                  |          swl:                    |
|            max:                  |            min:  0.03            |
|            mean:                 |            max:  0.05            |
|            base:                 |          swcr:                   |
|            stddev:               |            min:  0.09            |
|            distribution:         |            max:  0.15            |
|            low_optimistic:       |          sorw:                   |
|          swl:                    |            min:                  |
|            <same as for swirr>   |            max:                  |
|          swcr:                   |          nw:                     |
|            <same as for swirr>   |            min:                  |
|          sorw:                   |            max:                  |
|            <same as for swirr>   |          now:                    |
|          krwend:                 |            min:                  |
|            <same as for swirr>   |            max:                  |
|          kroend:                 |          krwend:                 |
|            <same as for swirr>   |            min:                  |
|          no:                     |            max:                  |
|            <same as for swirr>   |          kroend:                 |
|          now:                    |            min:                  |
|            <same as for swirr>   |            max:                  |
|          sorg:                   |                                  |
|            <same as for swirr>   |                                  |
|          sgcr:                   |                                  |
|            <same as for swirr>   |                                  |
|          ng:                     |                                  |
|            <same as for swirr>   |                                  |
|          nog:                    |                                  |
|            <same as for swirr>   |                                  |
|          krgend:                 |                                  |                
|            <same as for swirr>   |                                  |
+----------------------------------+----------------------------------+

scheme
  The scheme parameter decides how many sets of relative permeability curves to generate as
  input to Flow. There are three options. With **shceme: global** only one set of relative 
  permeability curves will be generated, and applied to all flow tubes in the model. With
  **shceme: individual** all flow tubes in the model will have its own set of relative permeability
  curves. With **scheme: regions_from_sim** FlowNet will extract the SATNUM regions from the 
  input model provided, and assign the same set of relative permeability curves to all flow tubes 
  that are (mostly) located within the same SATNUM region. The default value is global.

interpolate
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
  if **interpolate** is set to **True** and the model has three active phases, this parameter will
  decide whether or not the interpolation for water/oil relative permeability and gas/oil relative 
  permeability will be performed independently. The default value is False.
  
  
regions
  This is a list where each list element will contain information about the saturation endpoints 
  and relative permeability endpoints within one SATNUM region, in addition to a region identifier. The 
  endpoints are shown in two figures below for clarification.
  The number of list elements needs to be equal to the number of SATNUM regions in the model,
  unless one of the regions is defined with identifier *None*. 
  
  id
    Region identifier. Default value is None.
  swirr
    The irreducible water saturation.
  swl
    Connate water saturation.
  swcr
    Critical water saturation
  sorw
    Residual oil saturation (that cannot be displaced by water)
  krwend
    Maximum relative permeability for water
  kroend
    Maximum relative permeability for oil
  nw, now, ng, nog
    Exponents in Corey parametrization
  sorg
    Residual oil saturation (that cannot be displaced by gas)
  sgcr
    Critical gas saturation
  krgend
    Maximum relative permeability for gas
  
  A water/oil model needs *swirr*, *swl*, *swcr*, *sorw*, *nw*, *now*, *krwend* and *kroend* to be defined.
  An oil/gas model needs *swirr*, *swl*, *sgcr*, *sorg*, *ng*, *nog*, *krgend* and *kroend* to be defined.
  A three phase model needs all 13 relative permeability parameters to be defined.


  
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


Equilibration
-------------



+----------------------------------+----------------------------------+
| Available options in config yaml | Example of usage                 |
+----------------------------------+----------------------------------+
|                                  |                                  |
| .. code-block:: yaml             | .. code-block:: yaml             |
|                                  |                                  |
|  flownet:                        |  flownet:                        |
|    model_parameters:             |    model_parameters:             |
|      equil:                      |      equil:                      |
|        scheme:                   |        scheme: global            |
|        regions:                  |         regions:                 |
|          id:                     |           id: None               |
|          datum_depth:            |           datum_depth:           |
|          datum_pressure:         |           datum_pressure:        |
|            min:                  |             min:                 |
|            max:                  |             max:                 |
|            mean:                 |           owc_depth:             |
|            base:                 |             min:                 |
|            stddev:               |             max:                 |
|            distribution:         |           goc_depth:             |
|          owc_depth:              |             min:                 |
|            min:                  |             max:                 |
|            max:                  |           id: 1                  |
|            mean:                 |           datum_depth:           |
|            base:                 |           datum_pressure:        |
|            stddev:               |             min:                 |
|            distribution:         |             max:                 |
|          goc_depth:              |           owc_depth:             |
|            same as for owc_depth |             min:                 |
|          gwc_depth:              |             max:                 |
|            same as for owc_depth |           goc_depth:             |
|				   |	         min:                 |
|				   |	         max:                 |
|                                  |                                  |
+----------------------------------+----------------------------------+

scheme
  The scheme parameter decides how many equilibration regions to generate as
  input to Flow. There are three options. With **shceme: global** the model will only have one  
  equilibration region, and applied to all flow tubes in the model. With
  **shceme: individual** all flow tubes in the model will act as its own equilibration region. 
  With **scheme: regions_from_sim** FlowNet will extract the EQLNUM regions from the 
  input model provided, and assign equilibraion regions to all flow tubes accordingly. 
  The default value is global.

regions
  This is a list where each list element will contain information about the datum depth, datum pressure and 
  fluid contacts within one equilibration region, in addition to a region identifier.
  The number of list elements needs to be equal to the number of EQLNUM regions in the model,
  unless one of the regions is defined with identifier *None*. 
  
  id
    Region identifier. Default value is None.
  datum_depth:
    Datum or reference depth in the equilibrium region.
  datum_pressure:
    Datum or reference pressure in the equilibrium region.
  owc_depth:
    Depth of the oil/water contact in the equilibrium region.
  goc_depth:
    Depth of the gas/oil contact in the equilibrium region.
  gwc_depth:
    Depth of the gas/water contact in the equilibrium region.

  The *datum depth* is just a number. The *datum pressure* and the different contacts 
  should be entered with a prior probability distribution.

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


Assisted history matching example
=================================

.. literalinclude:: ../examples/norne_parameters.yml
   :language: yaml
   :linenos:

Prediction example
==================

.. literalinclude:: ../examples/norne_pred.yml
   :language: yaml
   :linenos:
