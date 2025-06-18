#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:33:22 2024

@author: bmcleod
"""

import numpy as np
import astropy.io.fits as fits
#import pyds9

#%%
folder = '/data/piper0/bmcleod/2025-02-TAOS2Data/taosdata/'

#ds9 = pyds9.DS9()
#%%

#250115/site1_cmos9_fseq06_1820.fits

date = '250115'
site = 1
seqname = 'fseq06'
darkname = 'dark_2s_03_04'

import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process observation metadata.")
    
    parser.add_argument('--date', type=str, default='250115',
                        help="Observation date in YYMMDD format (default: '250115')")
    
    parser.add_argument('--site', type=int, default=1,
                        help="Site number identifier (default: 1)")
    
    parser.add_argument('--seqname', type=str, default='fseq06',
                        help="Sequence name (default: 'fseq06')")
    
    parser.add_argument('--darkname', type=str, default='dark_2s_03_04',
                        help="Dark frame name (default: 'dark_2s_03_04')")

    parser.add_argument('--foc_min', type=int, default=1820,
                        help="Minimum focus value (default: 1820)")

    parser.add_argument('--foc_max', type=int, default=2025,
                        help="Maximum focus value (default: 2025)")

    parser.add_argument('--foc_delta', type=int, default=5,
                        help="Focus step size (default: 5)")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()


date = args.date
site = args.site
seqname = args.seqname
darkname = args.darkname
foc_min = args.foc_min
foc_max = args.foc_max
foc_delta = args.foc_delta

#for imnum in range(2558,2589, 2):
#for imnum in range(1820,2025,5):
for imnum in range(foc_min, foc_max+1, foc_delta):
    for chipnum in range(10):
        filename = folder + f'{chipnum}/{date}/site{site}_cmos{chipnum}_{seqname}_{imnum}.fits'
        print(filename)
        data = fits.getdata(filename)
        darkfilename = folder +f'{chipnum}/{date}/site{site}_cmos{chipnum}_{darkname}.fits'
        dark = fits.getdata(darkfilename)
        if chipnum==0:
            xsize = data.shape[1]
            ysize = data.shape[0]
            output_array = np.zeros((ysize*2, xsize*5))
            
        data = data - dark + 200
        
        data = data - np.median(data, axis=1, keepdims=True)

        if chipnum<5:
            ymin=0
            ymax=ysize
            xmin = chipnum * xsize
            xmax = xmin + xsize
            print(chipnum, xmin, xmax, ymin, ymax, output_array.shape)
            output_array[ymin:ymax, xmin:xmax] = data
        else:
            ymin = ysize
            xmin = xsize * (9 - chipnum)
            ymax = ymin + ysize
            xmax = xmin + xsize
            print(chipnum, xmin, xmax, ymin, ymax, output_array.shape)
            output_array[ymin:ymax, xmin:xmax] = np.flip(data)
            

    os.makedirs(folder + f'mosaics/{date}', exist_ok = True)
    outname =  folder + f'mosaics/{date}/site{site}_mosaic_{seqname}_{imnum}.fits'
    fits.writeto( outname, output_array,overwrite=True)
    
    
    
    
    
    
