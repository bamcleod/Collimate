# Collimate

Software for collimating wide-field telescopes.

## Description

This software package allows the user to analyze defocused images to
determine the dependence of low-order aberrations with field position.
These coeffiecents are then transformed into required motions of the
secondary mirror to bring the telescope into collimation.

## History

The underlying tools were written over 25 years ago and have been used
to keep the Mt. Hopkins 1.2m telescope collimated.  More recently a
Python interface was developed to be used for collimating the TAOSII
telescopes at San Pedro Martir.

## Getting Started

### Dependencies

* This software currently runs under Linux only as it contains a pre-compiled program.
* It requires having the following python packages installed:
  * numpy
  * matplotlib
  * pyds9
  * astropy
  * yaml


### Installing

git clone https://github.com/bamcleod/Collimate

Modify the programs make_taos_mosaic.py and summarize_taos_data_sets.py to point to the correct data folder.  This folder should be one that has subfolders named "0" through "9".

### Executing the programs (TAOSII specific)

* Optional: find the data sets.
```
python summarize_taos_data_sets.py
```
Sample output:
```

python make_taos_mosaics.py --date 250115 --site 1 --seqname fseq06 --foc_min 1820 --foc_max 2015 --foc_delta 5; # range: 195
python make_taos_mosaics.py --date 250115 --site 2 --seqname fseq04 --foc_min 2068 --foc_max 2088 --foc_delta 2; # range: 20
python make_taos_mosaics.py --date 250121 --site 3 --seqname fseq07 --foc_min 2555 --foc_max 2575 --foc_delta 2; # range: 20
```

* The first step is to merge the multiple focal plane sensors into a single large image using the program make_taos_mosaic.py  This program also performs dark subtraction while doing the merge.

   To each line you will need to append the appropriate dark file name, e.g.
```
python make_taos_mosaics.py --date 250115 --site 2 --seqname fseq04 --foc_min 2068 --foc_max 2088 --foc_delta 2 --darkname dark002s_018

```

* Now you can run the main Jupyter notebook, Collimate_TAOS2_Site2.ipynb, again changing the root folder appropriately.



## Authors

Brian McLeod

