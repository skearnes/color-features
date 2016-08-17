#!/bin/bash

#PBS -l nodes=1:ppn=1,walltime=24:00:00
#PBS -j oe

echo REF $ref
echo FIT $fit
echo PREFIX $prefix

echo PYTHON `which python`

cd $PBS_O_WORKDIR
time python ~/git/color-features/oe_utils/scripts/rocs.py -r $ref -f $fit -o ${prefix}-rocs.h5
