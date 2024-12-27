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

    # Set up the figure and subplots
    fig, axes = plt.subplots(1, len(valid_images), figsize=(6 * len(valid_images), 6))

    # Ensure axes is a list for consistent iteration
    if len(valid_images) == 1:
        axes = [axes]

    # Plot each image
    for i, (ax, (img, title)) in enumerate(zip(axes, valid_images)):
        im = ax.imshow(img, cmap='inferno')
        if verbosity >= 1:
            ax.set_title(title, fontsize=14)
        if verbosity >= 2:
            # Add individual color bars for each plot
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Intensity", fontsize=10)
        if verbosity >= 3:
            # Include axes
            ax.set_xlabel('X-axis (pixels)')
            ax.set_ylabel('Y-axis (pixels)')
        else:
            ax.axis('off')  # Turn off axes for verbosity < 3

    if verbosity >= 1:
        # Add a fixed main title
        fig.suptitle("SPHEREx Image Visualization", fontsize=16, y=1.02)

    # Adjust subplot spacing
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for color bars and main title
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