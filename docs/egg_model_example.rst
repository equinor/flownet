==========================
The Egg model example
==========================

The "Egg Model" is a synthetic reservoir model.

Data files
==========

- `The Egg Model - data files <https://data.4tu.nl/articles/The_Egg_Model_-_data_files/12707642>`_


Simulations settup
==================

All vertical wells in the egg model has the same depth. This creates degenerates 3D triangles in the triangulation algorithim used to generate the flow network.
This can be solved by changing the depth of he wells in COMPAT section of the egg model simulaiton fila: `Egg_Model_ECL.DATA`. 

.. code-block::
    :language: yaml    
        COMPDAT
            'INJECT1'    2*    4     7 'OPEN' 2*     0.2 	1*          0 / 
            'INJECT2'    2*    4     7 'OPEN' 2*     0.2 	1*          0 /
            'INJECT3'    2*    4     7 'OPEN' 2*     0.2 	1*          0 /
            'INJECT4'    2*    4     7 'OPEN' 2*     0.2 	1*          0 /
            'INJECT5'    2*    4     7 'OPEN' 2*     0.2 	1*          0 /
            'INJECT6'    2*    4     7 'OPEN' 2*     0.2 	1*          0 /
            'INJECT7'    2*    4     7 'OPEN' 2*     0.2 	1*          0 /
            'INJECT8'    2*    4     7 'OPEN' 2*     0.2 	1*          0 /
            'PROD1'      2*    1     3 'OPEN' 2*     0.2 	1*          0 / 
            'PROD2'      2*    1     3 'OPEN' 2*     0.2 	1*          0 / 
            'PROD3'      2*    1     3 'OPEN' 2*     0.2 	1*          0 / 
            'PROD4'      2*    1     3 'OPEN' 2*     0.2	1*          0 / 
        /


The configuration files follows the `YAML standard <https://yaml.org/>`_.

Assisted history matching example
=================================

.. literalinclude:: ../examples/norne_parameters.yml
   :language: yaml
   :linenos:
