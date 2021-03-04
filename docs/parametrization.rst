Prior distributions in FlowNet
===========================================

The different parameters to be tuned are defined in the **model_parameters** 
section of the FlowNet config yaml. At present, the model can be parameterized 
with the following parameters:

* Permeability
* Porosity
* Bulk volume multiplier
* Fault multiplier
* Saturation endpoints and relative permeability endpoints
* Datum pressures and contacts
* Aquifer size (relative to the bulk volume in the model)


All parameters need an initial guess 
on what values they can take. This is referred to as the prior probability distribution.

The following keys are available for defining the different prior distributions

distribution
  The type of probability distribution. If a model parameter is included, the distribution
  type must be defined. The other parameters below are included depending on choice of
  distribution. The default value distribution type is *uniform*.

min
  Depending on modelling choices, this parameter can have different meanings.
  The parameter can act as the minimum value of the chosen prior probability distribution,
  or as the low value of a relative permeability
  modelling parameter when using the SCALrecommendation option in pyscal. 
  
max
  Depending on modelling choices, this parameter can have different meanings.
  The parameter can act as the maximum value of the chosen prior probability distribution,
  or as the high value of a relative permeability
  modelling parameter when using the SCALrecommendation option in pyscal.

base
  Depending on modelling choices, this parameter can have different meanings.
  It can act as the mode of the prior probability distribution in a triangular distribution,
  or as the constant value in a Dirac distribution, or the *base* value of a relative permeability
  modelling parameter when using the SCALrecommendation option in pyscal.
  
mean
  The mean or expected value of the prior probability distribution

stddev
  The standard deviation of the prior probability distributions

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
| Dirac (constant)          | const            |      |      |      |  x   |      |        
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

        

Saturation endpoints and relative permeability endpoints
--------------------------------------------------------

FlowNet uses `pyscal <https://github.com/equinor/pyscal>`_ for generating relative permeability input curves for Flow. 
For detailed documentation on pyscal, read the `pyscal documentation <https://equinor.github.io/pyscal>`_. This text 
will only describe how FlowNet uses pyscal.

pyscal can parameterize curves using either Corey parameters or LET parameters. 
FlowNet only accepts Corey parameters as input at this point.


The input related to relative permeability modelling has its own section in the config yaml file,
where the following parameters can be defined. 

scheme
  The scheme parameter decides how many sets of relative permeability curves to generate as
  input to Flow. There are three options. With **shceme: global** only one set of relative 
  permeability curves will be generated, and applied to all flow tubes in the model. With
  **shceme: individual** all flow tubes in the model will have its own set of relative permeability
  curves. With **scheme: regions_from_sim** FlowNet will extract the SATNUM regions from the 
  input model provided, and assign the same set of relative permeability curves to all flow tubes 
  that are (mostly) located within the same SATNUM region. The default value is global.

interpolate
  pyscal has an option to use SCALrecommendation. This is due to the fact that SCAL experts often
  will provide three sets of relative permeability curves (one pessimistic set , one base set and 
  one optimistic set) to run sensitivities on a reservoir model. This introduces the option of 
  generating new sets of relative permeability curves within the envelope created by the low/bas/high 
  sets of curves by using an interpolation parameter (potentially two interpolation parameters in three
  phase models). This will limit the number of history matching parameters, especially when the number 
  of SATNUM regions is large. The default value is False.

independent_interpolation
  if **interpolate** is set to **True** and the model has three active phases, this parameter will
  decide whether or not the interpolation for water/oil relative permeability and gas/oil relative 
  permeability will be performed independently. The default value is False.
  
  
regions
  This is a list where each list elements will contain information about the saturation endpoints 
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
|          id:                     |            min:                  |
|          swirr:                  |            max:                  |
|            min:                  |          swl:                    |
|            max:                  |            min:                  |
|            mean:                 |            max:                  |
|            base:                 |          swcr:                   |
|            stddev:               |            min:                  |
|            distribution:         |            max:                  |
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


  
    


  To limit the number of history 
  matching parameters, FlowNet provides the option to 
  interpolate between three sets of relative permeability curves. This way each SATNUM region will 
  only have one history matching parameter (possibly two if oil/gas and water/oil are 
  interpolated independently). This option is selected by setting this **interpolate** 
  option to **True**. The default value is False.

.. figure:: https://equinor.github.io/pyscal/_images/gasoil-endpoints.png
  
   Visualization of the gas/oil saturation endpoints and gas/oil relative permeability endpoints as modelled by pyscal. 

.. figure:: https://equinor.github.io/pyscal/_images/wateroil-endpoints.png
  
   Visualization of the water/oil saturation endpoints and water/oil relative permeability endpoints as modelled by pyscal. 


When using the interpolation option for relative permeability, some of the keywords above 
have a different meaning. This applies to **min**, **base**, and **max**. There is also an
additional keyword **low_optimistic** which only is meaningful to define for relative permeability.

Each of the input parameters needs a low, base, and high value to be defined. This is done through
the **min** (low), **base** and **max** (high) keywords. 
For some parameters a low numerical value is favorable. This can be indicated by setting 
**low_optimistic** to **True** for that parameter (the default value of low_optimistic is False).

The SCALrecommendation 
option in pyscal takes three values for each of the input parameters to create
three sets of input curves, later used as an envelope to interpolate between. 

There will be one *pessimistic*
set of curves, consisting of the low values supplied in the config file (this will be the *min* 
values, unless *low_optimistic* is set to *True*), one *optimistic* set of curves, consisting of
the high values supplied in the config yaml file (this will be the *max* values, unless *low_optimistic*
is set to *True*), and one *base* set of curves using the *base* values supplied.

pyscal will generate an interpolation parameter (two if **independent_interpolation** is set to **True**)
going from -1 (representing the pessimistic curve set) to 1 (representing the optimistic curve set).
FlowNet will pass this interpolation parameter to ERT for history matching, instead of the individual 
saturation endpoint or relative permeability endpoint parameters.
