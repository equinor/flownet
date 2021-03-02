Prior distributions in FlowNet
===========================================

The different parameters to be tuned are defined in the FlowNet config yaml.
Here, the choices available for the following parameters will be explained in more detail :ref:`porosity and pore volume<porosity-and-pore-volume>`

* Aquifers_
* Equilibration_ parameters
* Faults_
* Permeabilities_
* 
  

All parameters needs an initial guess on what values they can take. 
This is referred to as the prior probability distribution.

There are several different prior distributions or data types defined for ERT. 
The following ones are available in Flownet:

:Normal distribution: The normal distribution is defined by the mean value and the standard deviation

:Log-normal distribution: The log-normal distribution in defined by the mean value and the standard deviation


.. _Aquifers:
.. _Equilibration:
.. _Faults:
.. _Permeabilities:
.. _porosity-and-pore-volume: 




The following keys are available for defining the different prior distributions

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


+------------+-------+------+------+------+------+------+
| Distribution type  | Min  | Max  | Mean | Base |Stddev|
+============+=======+======+======+======+======+======+
| Normal             |      |      |   x  |      |   x  |        
+--------------------+------+------+------+------+------+
  

When using the interpolation option for relative permeability, some of the keywords above 
have a different meaning. This applies to **min**, **base**, and **max**. There is also an
additional keyword **low_optimistic** which only is meaningful for relative permeability.

Each of the input parameters needs a low, base, and high value to be defined. This is done through
the **min** (low), **base** and **max** (high) keywords. 
For some parameters a low numerical value is favorable. This can be indicated by setting 
**low_optimistic** to **True** for that parameter (the default value of low_optimistic is False).



The SCALrecommendation 
option in pyscal takes three values for each of the input parameters to create
three sets of input curves, later used as an envelope to interpolate between. 



There will be one *pessimistic*
set of curves, consisting of the low values supplied in the config file (this will be the *min* 
values, unless *low_optimistic* is set to *True*), one *optimistic* set of curves, consiting of
the high values supplied in the config yaml file (this will be the *max* values, unless *low_optimistic*
is set to *True*), and one *base* set of curves using the *base* values supplied.