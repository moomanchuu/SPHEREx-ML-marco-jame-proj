import astropy
from astropy.io import fits

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import binned_statistic_2d
from scipy.ndimage import convolve
from scipy.signal import correlate

import matplotlib as mpl


from scipy.signal import correlate
from astropy.io import fits


def calculate_original_image_pixels(sphereX_pixels, resolution_ratio=9, psf_length=54):
    """
    Calculate the required original image pixel dimensions based on SPHEREx pixel size.

    """
    return sphereX_pixels * resolution_ratio + 2 * psf_length - 1

# sphereX_pixels_list = [16,32,64,128, 256, 512, 1024,2048]
# original_image_sizes = {px: calculate_original_image_pixels(px) for px in sphereX_pixels_list}
# for px, orig in original_image_sizes.items():
#     print(f"SPHEREx Pixels: {px}, Original Image Pixels: {orig}")


def processImg(raw_image, psf_file='SPHEREx/psf_data/simulated_PSF_2DGaussian_1perarray.fits', psf_length=54):
    """
    Processes an image for SPHEREx by convolving it with a PSF and binning the result.


    """
    # Open the FITS file and extract PSF data
    hdul = fits.open(psf_file)
    hdu_psf = hdul[42]
    reso_ratio = int(3.1 / hdu_psf.header['HIERARCH platescale'])
    psf = hdu_psf.data
    hdul.close()

    # Crop and normalize the PSF
    c = int(psf.shape[0] / 2)
    psf = psf[c - psf_length:c + psf_length, c - psf_length:c + psf_length]
    psf = psf / np.sum(psf)

    # Convolve raw image with PSF
    convolved = correlate(raw_image, psf, mode='valid')

    # Bin the image
    def bin2d(img, ratio):
        m_bins = img.shape[0] // ratio
        n_bins = img.shape[1] // ratio
        img_binned = img.reshape(m_bins, ratio, n_bins, ratio).sum(3).sum(1)
        img_binned /= ratio**2
        return img_binned

    binned = bin2d(convolved, reso_ratio)

    return convolved, binned

import matplotlib.pyplot as plt

def plotImg(raw_image, convolved=None, binned=None, verbosity=1):
    """
    Plots the raw image, and optionally the convolved and binned images, with verbosity levels.
    
    Parameters:
    - raw_image (2D array): Raw image to plot.
    - convolved (2D array, optional): Convolved image to plot. Default is None.
    - binned (2D array, optional): Binned image to plot. Default is None.
    - verbosity (int, optional): Level of detail to include in the plot. Default is 1.
      - 0: Just the images.
      - 1: Add main title and subtitles for each plot.
      - 2: Include individual color bars.
      - 3: Include axes with labels.
    """
    images = [raw_image, convolved, binned]
    titles = ["Raw Image", "Convolved Image", 'Binned Image (3.1"/pix)']

    # Filter out None images
    valid_images = [(img, title) for img, title in zip(images, titles) if img is not None]

    fig, axes = plt.subplots(1, len(valid_images), figsize=(6 * len(valid_images), 6))

    if len(valid_images) == 1:
        axes = [axes]

    for i, (ax, (img, title)) in enumerate(zip(axes, valid_images)):
        im = ax.imshow(img, cmap='inferno')
        if verbosity >= 1:
            ax.set_title(title, fontsize=14)
        if verbosity >= 2:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Intensity", fontsize=10)
        if verbosity >= 3:
            # Include axes
            ax.set_xlabel('X-axis (pixels)')
            ax.set_ylabel('Y-axis (pixels)')
        else:
            ax.axis('off')  

    if verbosity >= 1:
        fig.suptitle("SPHEREx Image Visualization", fontsize=16, y=1.02)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) #Prob should adjust later tbh
    plt.show()


import matplotlib.pyplot as plt

def plot_spherex_data(indices, is_lensed, simulated_arrays, spherex_arrays, indice):
    """
    Plot SPHEREx data for a given index.

    """
    row_index = list(indices).index(indice)

    lensed_status = "True" if is_lensed[row_index] == 1 else "False"
    simulated_array = simulated_arrays[row_index]
    spherex_array = spherex_arrays[row_index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"SPHEREx Image Visualization\nIndex = {indice}, Lensed = {lensed_status}", fontsize=16)

    axes[0].imshow(simulated_array, cmap="inferno")
    axes[0].set_title("Original Simulated Image")
    axes[0].axis("off")

    axes[1].imshow(spherex_array, cmap="inferno")
    axes[1].set_title("SPHEREx Resolution Image")
    axes[1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()










#Legacy Code
'''

hdul =fits.open('SPHEREx\psf_data\simulated_PSF_2DGaussian_1perarray.fits')
hdu_psf = hdul[42]
reso_ratio = int(3.1/hdu_psf.header['HIERARCH platescale'])

# Crop PSF image to 98x98 pixels, then normalize the PSF so sum of all pixel = 1
psf = hdu_psf.data
c = int(psf.shape[0]/2)
psf_length = 54 #49? Was told to keep it at 54
psf = psf[c-psf_length:c+psf_length, c-psf_length:c+psf_length]
psf = psf/np.sum(psf)



hdul =fits.open('psf_data\simulated_PSF_2DGaussian_1perarray.fits')
hdu_psf = hdul[42]
reso_ratio = int(3.1/hdu_psf.header['HIERARCH platescale'])

# Crop PSF image to 98x98 pixels, then normalize the PSF so sum of all pixel = 1
psf = hdu_psf.data
c = int(psf.shape[0]/2)
psf_length = 54 #49? Was told to keep it at 54
psf = psf[c-psf_length:c+psf_length, c-psf_length:c+psf_length]
psf = psf/np.sum(psf)

#n_pix = n_spherex_pix * reso_ratio + 2 * psf_length - 1

# Convolve raw image with psf with scipy.signal.correlate
convolved = correlate(raw, psf, mode='valid')

# Define custom bin function
def bin2d(img, ratio):
    m_bins = img.shape[0]//ratio
    n_bins = img.shape[1]//ratio
    img_binned = img.reshape(m_bins, ratio, n_bins, ratio).sum(3).sum(1)
    img_binned /= ratio**2
    return img_binned

binned = bin2d(convolved, reso_ratio)
'''