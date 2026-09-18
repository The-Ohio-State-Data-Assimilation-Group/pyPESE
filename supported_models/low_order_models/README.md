# Generic pyPESE functions for low-order models
> Written by Man-Yau (Joseph) Chan

## Description

This directory contains some generic functions to evoke pyPESE for data samples 
based on low-order models.

To be clear, a low-order model here is defined as any model whose state vectors 
contain less than 1,000 elements. Such models include the Lorenz models (1963, 
1996 & 2005).


## Important Notes

1)  Unlike the pyPESE for high-order models (e.g., CAM), I am not providing self-
    contained parallelized programs to run pyPESE. Instead I am providing 
    functions that you will need to import into your python workflow.

2)  The Gaussian resampling in Chan et al (2022) is only computationally 
    efficient when the number of ensemble members/data samples (Ne) is much 
    less than the number of state vector elements (Nx; i.e., Ne << Nx). In the
    situation where Ne >> Nx, the traditional approach to Gaussian resampling 
    is more efficient. The functions contained here automatically switches 
    between both kinds of Gaussian resampling.






