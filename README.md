# ROCS-Derived Features for Virtual Screening

This repository contains supporting code and data for http://arxiv.org/abs/1606.01822.

## Calculating _color components_ and _color atom overlaps_

* Install oe_utils: `python setup.py install` or `python setup.py develop`
* See the PBS scripts in `paper/code` for usage examples

## Data analysis

The data used for the analysis reported in http://arxiv.org/abs/1606.01822 is given in `paper/data/data.tar.gz`. To regenerate the data tables, run the following:

```bash
# Extract the data
tar -xzf paper/data/data.tar.gz

# Tanimoto data tables
for dataset in dude muv chembl; do
  python paper/code/analysis.py \
    --root data-tversky \
    --dataset_file paper/data/${dataset}-datasets.txt \
    --prefix ${dataset}
done

# Tversky data tables
for dataset in dude muv chembl; do
  python paper/code/analysis.py \
    --root data-tversky \
    --dataset_file paper/data/${dataset}-datasets.txt \
    --prefix ${dataset} \
    --tversky
done
```
