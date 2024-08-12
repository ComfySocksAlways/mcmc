#!/bin/bash

# inputs
N_SAMPLES=100000
VERBOSE_FLAG=""

if [ "$1" == "-v" ]; then
    VERBOSE_FLAG="--verbose"
fi

echo "n_samples: $N_SAMPLES"
# call script
python3 genome_plv2.py --n_samples $N_SAMPLES $VERBOSE_FLAG
