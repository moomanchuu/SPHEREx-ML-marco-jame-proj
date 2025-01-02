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

import numpy as np
import os

def addNoise(array, mode="constantdarkcurrent", parameter=None, template=None, seed=None):
    """
    Adds noise to a 2D NumPy array.

    Parameters:
    - array (2D array): The input 2D NumPy array to which noise will be added.
    - mode (str): The type of noise to add. Supported modes are:
        * "Gaussian": Gaussian noise with mean 0 and standard deviation `parameter`.
        * "constantdarkcurrent": Poisson noise with a constant mean `parameter`.
        * "darkcurrent": Poisson noise based on a template (spatially varying).
    - parameter (float): Parameter for the noise. For example:
        - For "Gaussian": Standard deviation (sigma).
        - For "constantdarkcurrent": Mean of Poisson distribution.
        - For "darkcurrent": Not used directly (template is used instead).
    - template (2D array): A 2D array for spatially varying dark current (used in "darkcurrent" mode).
    - seed (int): Seed for random number generation (optional). If None, uses a truly random seed.

    Returns:
    - noisy_array (2D array): The input array with added noise.
    """
    if seed is not None:
        np.random.seed(seed)
    else: #stupid MPI pepega 
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    
    # Add noise based on the mode
    if mode == "Gaussian":
        if parameter is None:
            raise ValueError("Parameter (standard deviation) must be specified for Gaussian noise.")
        noise = np.random.normal(loc=0, scale=parameter, size=array.shape)
    elif mode == "constantdarkcurrent":
        if parameter is None:
            raise ValueError("Parameter (mean) must be specified for constant dark current noise.")
        noise = np.random.poisson(lam=parameter, size=array.shape)
    elif mode == "darkcurrent":
        if template is None:
            raise ValueError("Template must be provided for dark current noise.")
        if template.shape != array.shape:
            raise ValueError(f"Template shape {template.shape} does not match array shape {array.shape}.")
        noise = np.random.poisson(lam=template)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'Gaussian', 'constantdarkcurrent', and 'darkcurrent'.")
    
    return array + noise

def plotImg(raw_image, convolved=None, binned=None, noisy_array=None, verbosity=1):
    """
    Plots the raw image, and optionally the convolved, binned, and noisy images, with verbosity levels.
    
    Parameters:
    - raw_image (2D array): Raw image to plot.
    - convolved (2D array, optional): Convolved image to plot. Default is None.
    - binned (2D array, optional): Binned image to plot. Default is None.
    - noisy_array (2D array, optional): Noisy image to plot. Default is None.
    - verbosity (int, optional): Level of detail to include in the plot. Default is 1.
      - 0: Just the images.
      - 1: Add main title and subtitles for each plot.
      - 2: Include individual color bars.
      - 3: Include axes with labels.
    """
    images = [raw_image, convolved, binned, noisy_array]
    titles = [
        "Raw Image",
        "Convolved Image",
        'Binned Image (3.1"/pix)',
        "Dark Current Added"
    ]

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
            ax.set_xlabel('X-axis (pixels)')
            ax.set_ylabel('Y-axis (pixels)')
        else:
            ax.axis('off')  

    if verbosity >= 1:
        fig.suptitle("SPHEREx Image Visualization", fontsize=16, y=1.02)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  
    plt.show()


def plotSPHERExData(indices, is_lensed, simulated_arrays, spherexBinnedArrays=None, spherexDC_arrays=None, indice=None):
    """
    Plot SPHEREx data for a given index, with optional binned and noisy images.

    Parameters:
    - indices (list): List of indices corresponding to the data.
    - is_lensed (list): Binary list indicating if the index corresponds to a lensed image (1) or not (0).
    - simulated_arrays (list): List of original simulated images.
    - spherexBinnedArrays (list, optional): List of SPHEREx resolution (binned) images. Default is None.
    - spherexDC_arrays (list, optional): List of SPHEREx images with dark current noise added. Default is None.
    - indice (int, optional): The index to visualize. If None, no plot is produced.
    """
    if indice is None:
        raise ValueError("An index (indice) must be specified for plotting.")

    # Get the row index for the given indice
    row_index = list(indices).index(indice)

    # Check lensed status
    lensed_status = "True" if is_lensed[row_index] == 1 else "False"

    images = [
        simulated_arrays[row_index],
        spherexBinnedArrays[row_index] if spherexBinnedArrays is not None else None,
        spherexDC_arrays[row_index] if spherexDC_arrays is not None else None
    ]
    titles = [
        "Original Simulated Image",
        "SPHEREx Resolution (Binned) Image",
        "Dark Current Added"
    ]

    # Filter out "None"
    valid_images = [(img, title) for img, title in zip(images, titles) if img is not None]

    fig, axes = plt.subplots(1, len(valid_images), figsize=(6 * len(valid_images), 6))
    fig.suptitle(f"SPHEREx Image Visualization\nIndex = {indice}, Lensed = {lensed_status}", fontsize=16)

    if len(valid_images) == 1:
        axes = [axes]

    for ax, (img, title) in zip(axes, valid_images):
        im = ax.imshow(img, cmap="inferno")
        ax.set_title(title, fontsize=14)
        ax.axis("off")

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