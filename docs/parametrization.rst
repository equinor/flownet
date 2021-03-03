Prior distributions in FlowNet
===========================================

The different parameters to be tuned are defined in the FlowNet config yaml.
All parameters needs an initial guess on what values they can take. 
This is referred to as the prior probability distribution.

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



Prior distributions when using the interpolation option for relative permeability
---------------------------------------------------------------------------------


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
