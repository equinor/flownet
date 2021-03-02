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
  
  

When using the interpolation option for relative permeability, some of the keywords above 
have a different meaning. This applies to *min*, *max*, and *base*. The SCALrecommendation 
option in pyscal takes in three values for each of the input parameters, used to create
three sets of input curves as an envelope to interpolate between. There will be one *pessimistic*
set of curves, consisting of the low values supplied in the config file (this will be the *min* 
values, unless *low_optimistic* is set to *True*), one *optimistic* set of curves, consiting of
the high values supplied in the config yaml file (this will be the *max* values, unless *low_optimistic*
is set to *True*), and one *base* set of curves using the *base* values supplied.