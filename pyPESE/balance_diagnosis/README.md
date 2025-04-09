# Functions to diagnose balanced states

&nbsp; &nbsp;

## Description
PESE-GC might generate virtual members with highly unbalanced states. 
This directory contains functions to estimate balanced states. 

UNDER DEVELOPMENT. THESE FUNCTIONS ARE NOT USED AT THE MOMENT.

&nbsp; &nbsp;




## Available sub-packages

&nbsp; 

### 1. `geostrophic_flow`
Given thermodynamic data on an evenly-sampled latitude-longitude global grid, 
this subpackage diagnoses geostrophically-balanced eastward flow velocities 
and geostrophically-balanced northward flow velocities.


Note that if the thermodynamic data is not global or not evenly-sampled, this
subpackage will not work. 

&nbsp; 



### 2. `geostrophic_height`
This subpackage computes geopotential heights that are geostrophically-balanced
with the non-divergent part of the horizontal flow velocities. 

The inputted flow velocities must be defined on an evenly-sampled lat-lon global
grid. The outputted geostrophic heights are defined on the same grid.