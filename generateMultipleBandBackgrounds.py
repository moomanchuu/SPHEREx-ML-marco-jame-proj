import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy
from astropy.table import Table
from astropy.io import fits
import h5py


import sys
from pathlib import Path

base_dir = Path.cwd()
spherex_path = base_dir / "SPHEREx"
sys.path.append(str(spherex_path))
import SPHERExScripts as SPHEREx_old  # from SPHEREx

print('generateMultipleBandBackgrounds.py has been executed....')
sourceFile = '/mnt/md127/SPHEREx/largeOutputs/spherex_flux_10binned.fits' #Change this as needed, this file is only for Rabbit
sourceCatalog = Table.read(sourceFile) 

SNRFilter = 5
filteredCatalog = sourceCatalog[sourceCatalog['snr_per_filter'] >= SNRFilter]
print(f'SNR Filter of {SNRFilter} applied, {len(filteredCatalog)} sources remain')

psfData = '/home/marco/SPHEREx-ML/SPHEREx-ML-marco-jame-proj/SPHEREx/psf_data/simulated_PSF_2DGaussian_1perarray.fits'
targetResolution = 6.2


def loadPSFData(filePath = '/home/marco/SPHEREx-ML/SPHEREx-ML-marco-jame-proj/SPHEREx/psf_data/simulated_PSF_2DGaussian_1perarray.fits',target_resolution = 6.2):
    hdul =fits.open(filePath)
    hdu_psf = hdul[42]
    reso_ratio = int(target_resolution/hdu_psf.header['HIERARCH platescale'])

    # Crop PSF image to 98x98 pixels, then normalize the PSF so sum of all pixel = 1
    psf = hdu_psf.data
    c = int(psf.shape[0]/2)
    psf_length = 54 #49? Was told to keep it at 54
    psf = psf[c-psf_length:c+psf_length, c-psf_length:c+psf_length]
    psf = psf/np.sum(psf)
    return psf,reso_ratio,psf_length



psf,reso_ratio,psf_length = loadPSFData(psfData, 
                                        targetResolution
                                        )
print(f'Reso Ratio: {reso_ratio}')
print(f'PSF Length: {psf_length}')


print('Executing SPHEREx.parallel_canvas_generator()....')
raw_outputs = []
dataTab = sourceCatalog
ra = dataTab['ra']
dec = dataTab['dec']

band_names = [f'band{i}' for i in range(1, 11)] #10 bands of SPHEREx


for i, band in enumerate(band_names, start=1):
    print(f"Generating map for {band}...")

    flux_column = dataTab[band]*1e6 #Convert mJy to nJy
    #flux_column = flux_column / np.nanmax(flux_column)  # Normalize flux to [0,1]

    #Define Offsets (else random)
    ra_offset = 270
    dec_offset = 66

    #Actually generate image data
    rawTest = SPHEREx_old.parallel_canvas_generator(ra, 
                                            dec,
                                            flux_column, 
                                            ra_offset = ra_offset,
                                            dec_offset = dec_offset, 
                                            dataTable = dataTab, 
                                            reso_ratio = reso_ratio,
                                            psf_length = psf_length,
                                            n_spherex_pix=128,
                                            n_sections=16)

    raw_outputs.append(rawTest)

#Save the data
output_filename = "/mnt/md127/SPHEREx/largeOutputs/spherex_flux_10binned_backgroundsTest.h5"
with h5py.File(output_filename, "w") as hdf:
    hdf.create_dataset("raw_outputs", data=raw_outputs)

print(f"Data saved to {output_filename}")