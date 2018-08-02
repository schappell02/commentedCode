This directory contains code which has been commented, for rest of code, see code_backup directory

scStars.py - python code with classes around handling and analyzing stellar sample, including Bayesian analysis of late-type cusp. Interface with SQL is used to retrieve sample. Interface with C++ to run MultiNest, likelihood sampling.

sc_mn_FINAL.cpp - C++ code for running MultiNest (likelihood sampler) and the likelihood construction for the Bayesian analysis of late-type density profile.

checkFlux.pro - IDL code to compare flux between dark subtracted frame and final reduced frame/cube, there was concern that flux was not being conserved during reduction process of OSIRIS spectral images.
