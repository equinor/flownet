==========================
The Egg model example
==========================

The "Egg Model" is a synthetic reservoir model.

Data files
==========

- `The Egg Model - data files <https://data.4tu.nl/articles/The_Egg_Model_-_data_files/12707642>`_


Simulations settup pre-processing
=================================

1) All vertical wells in the egg model has the same depth. This creates degenerates 3D triangles in the triangulation algorithim used to generate the flow network.
This can be solved by changing the depth of he wells in COMPAT section of the egg model simulaiton fila: `Egg_Model_ECL.DATA`. 


ADD THIS FOLLOWING TEXT PROPERLLY:

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
        
        

2) The section PROPS should be save as a indepedend file call PROPS.inc.


            DENSITY
                900 1000          1 /
            PVCDO
                400          1 1.000E-05        5        0/
            PVTW
                400          1 1.000E-05        1        0/

                
3) Output variables of interest must be specify in this regards we add in the SUMMARY section
        SUMMARY
            FOPR
            FWPR
            FWIR
            WOPR
            'PROD1'
            'PROD2'
            'PROD3'
            'PROD4'
            /
            WWPR
            'PROD1'
            'PROD2'
            'PROD3'
            'PROD4'
            /
            WWIR
            'INJECT1' 
            'INJECT2'
            'INJECT3' 
            'INJECT4' 
            'INJECT5'
            'INJECT6'
            'INJECT7'
            'INJECT8'
            /   
            FOPT
            FWPT
            FWIT
            WLPR
            'PROD1'
            'PROD2'
            'PROD3'
            'PROD4'
            /
            WBHPH
            /
            WBHP
            /


4) Then you can run OPM FLOW to update


ADD BASH STYLE COMMAND
```bash
    flow Egg_Model_ECL.DATA
```

The configuration files follows the `YAML standard <https://yaml.org/>`_.

Assisted history matching example
=================================

.. literalinclude:: ../examples/norne_parameters.yml
   :language: yaml
   :linenos:
