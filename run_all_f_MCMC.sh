#!/bin/bash

# Script to execute all f_MCMC*.py files
for file in f_MCMC*.py; do
    echo "Running $file with IPython"
    ipython --matplotlib=agg -c "run '$file'"
done
