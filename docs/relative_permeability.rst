FlowNet uses `pyscal <https://github.com/equinor/pyscal>`_ for generating relative permeability input curves for Flow. pyscal can parameterise curves using either Corey parameters or LET parameters. FlowNet currently only accepts Corey parameters as input.

The input related to relative permeability modelling has its own section in the config yaml file. 


flownet:
  .
  .
  .
  model_parameters:
    .
    .
    .
    relative_permeability:
      scheme_.: global, regions_from_sim or interpolate
      interpolate:
      independent_interpolation:
      regions:
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
        

