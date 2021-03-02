FlowNet uses `pyscal <https://github.com/equinor/pyscal>`_ for generating relative permeability input curves for Flow. 
For detailed documentation on pyscal, read the `pyscal documentation <https://equinor.github.io/pyscal>`_. This text 
will only describe how FlowNet uses pyscal.

pyscal can parameterize curves using either Corey parameters or LET parameters. 
FlowNet only accepts Corey parameters as input at this point.


The input related to relative permeability modelling has its own section in the config yaml file. 


flownet:
  model_parameters:
    relative_permeability:
      scheme_.: 
      interpolate: 
      independent_interpolation:
|      regions:
        id:
        swirr:
        swl:
        swcr:
        sorw:
        krwend:
        kroend:
        no:
        now:
        sorg:
        sgcr:
        ng:
        nog:
        krgend:
          min:
          mean:
          max:
          base:
          stddev:
          distribution:
          low_optimistic:
          
          
          
.. _scheme:

Write about scheme here

:scheme: Determines how the relative permeability curves are generated. Scheme can be set to global, 
  regions_from_sim or individual. 
  **Global** will generate only one set of curves used for all flow tubes in the model. 
  **Individual** will generate one set of relative permeability curves for each flow tube in the model. 
  **Regions_from_sim** will generate on set of relative permeability curves for each 
  SATNUM region defined in the input simulation model. The default value is global.
:interpolate: To limit the number of history matching parameters, FlowNet provides the option to 
  interpolate between three sets of relative permeability curves. This way each SATNUM region will 
  only have one history matching parameter (possibly two if oil/gas and water/oil are 
  interpolated independently). This option is selected by setting this **interpolate** 
  option to **True**. The default value is False.
:independent_interpolation: If set to True, the interpolation of relative permeability curves for water/oil 
  is done independently to the interpolation for gas/water. Note that this can potentially lead to 
  generation of SWOF/SGOF tables that will not be accepted by Flow.
:regions:
  :id: Region identifier. Default value is **None**. 
    If there are no values specified for a specific region, 
    the values specified for region **None** will be used.
  :swirr: The irreducible water saturation
  :swl:
  :swcr:
  :sorw:
  :krwend:
  :kroend:
  :no:
  :now:
  :sorg:
  :sgcr:
  :ng:
  :nog:
  :krgend:

  All of the model parameters listed above needs to be defined by a prior probability distribution.
    :min: The minimum value of the distribution, or the low value used to build curves for interpolation
    :mean: The mean value of the distribution
    :max: The maximum value of the distribution, or the high value used to build curves for interpolation
    :base: The mode of the distribution, or the base value used to build curves for interpolation
    :stddev: The standard deviation of the prior distribution
    :distribution: The type of distribution. The available options are *uniform*, *logunif*, 
      *normal*, *lognormal*, *triangular*, *const*
    :low_optimistic: The interpolation option will interpolate between an optimistic 
      set of curves and a pessimistic set of curves. For some model parameters, the numerical *low* value
      should be used to generate the *optimistic* curve (and in turn used the numerical *high* 
      value to generate the *pessimistic* curve. If that is the case, set **low_optimistic** to True.

.. code-block:: yaml
  flownet:
    model_parameters:
      relative_permeability:
        scheme_.: 
        interpolate: 
        independent_interpolation:
|        regions:
          id:
          swirr:
          swl:
          swcr:
          sorw:
          krwend:
          kroend:
          no:
          now:
          sorg:
          sgcr:
          ng:
          nog:
          krgend:
            min:
            mean:
            max:
            base:
            stddev:
            distribution:
            low_optimistic:


