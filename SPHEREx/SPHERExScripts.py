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


from astropy.modeling.models import Sersic2D
from scipy.integrate import dblquad
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.stats import binned_statistic_2d
from tqdm import tqdm





#Function for Loading in SPHEREx PSF Data: 
def loadPSFData(filePath = 'SPHEREx\psf_data\simulated_PSF_2DGaussian_1perarray.fits'):
    hdul =fits.open(filePath)
    hdu_psf = hdul[42]
    reso_ratio = int(3.1/hdu_psf.header['HIERARCH platescale'])

    # Crop PSF image to 98x98 pixels, then normalize the PSF so sum of all pixel = 1
    psf = hdu_psf.data
    c = int(psf.shape[0]/2)
    psf_length = 54 #49? Was told to keep it at 54
    psf = psf[c-psf_length:c+psf_length, c-psf_length:c+psf_length]
    psf = psf/np.sum(psf)
    return psf,reso_ratio,psf_length



def add_sersic_to_canvas(canvas, x, y, flux, n=4, r_eff=10, ellip=0.5, theta=0, x_grid=None, y_grid=None): 

    """
    Add a Sersic profile to the canvas with the specified total flux.

    Parameters:
    canvas (2D array): The image canvas to draw on.
    x, y (int): The central position of the Sersic profile.
    flux (float): The total flux of the source.
    n (float): The Sersic index.
    r_eff (float): The effective radius of the Sersic profile.
    ellip (float): The ellipticity of the profile (0 is circular).
    theta (float): The position angle of the profile.
    x_grid, y_grid (2D arrays): Predefined grids for x and y coordinates (optional).
    """

    x_grid, y_grid = np.mgrid[:canvas.shape[0], :canvas.shape[1]] 

    # Create an initial Sersic2D model with amplitude 1 
    initial_sersic = Sersic2D(amplitude=1, r_eff=r_eff, n=n, ellip=ellip, theta=theta)
    
    # Shift the grid to center the Sersic profile at (x, y)
    x_grid_shifted = x_grid - x
    y_grid_shifted = y_grid - y
    
    # Evaluate the initial Sersic model on the shifted grid
    initial_profile = initial_sersic(x_grid_shifted, y_grid_shifted)
    
    # Scale the amplitude to match the desired total flux
    amplitude = flux / np.sum(initial_profile)
    
    # Create a Sersic2D model with the scaled amplitude 
    sersic = amplitude * initial_profile
    
    # Evaluate the Sersic model on the shifted grid and add to the canvas
    canvas += sersic

#Generate BackGround Image

from multiprocessing import Pool, cpu_count
from multiprocessing import Manager
import time
#V6: Removes most of the metadata generation, also hard codes it such that each image is of SPHEREx Pix = 2061 (3.1779 sq degrees)

def gen_globalmap_section_indexes(section_size,sections_per_side):

    # Initialize list to store the section position indices
    section_position_idxs = []

    # Loop through the base array and assign values to corresponding sectors in the expanded array
    for i in range(sections_per_side):
        for j in range(sections_per_side):
            # Get the value for the current section
            # Get the position indices for this section
            section_positions = []
            for row in range(i*section_size, (i+1)*section_size):
                for col in range(j*section_size, (j+1)*section_size):
                    section_positions.append((row, col))

            section_position_idxs.append(section_positions)

    return section_position_idxs

def add_sections_to_canvas(stitchedCanvas, populatedSection, section_positions):
    """
    Applies the values of populatedSection into the corresponding positions in stitchedCanvas.
    
    Args:
    stitchedCanvas: The large canvas where the values will be placed.
    populatedSection: The smaller section array to be applied.
    section_positions: The position indices in stitchedCanvas where the populatedSection should be applied.
    
    Returns:
    stitchedCanvas: The updated canvas with the populatedSection applied.
    """
    # Flatten the populatedSection so we can iterate over its values
    flat_populated_section = populatedSection.flatten()
    
    # Iterate over the section_positions and assign values from populatedSection
    for idx, (row, col) in enumerate(section_positions):
        stitchedCanvas[row, col] = flat_populated_section[idx]
    
    return stitchedCanvas

def add_sersic_to_canvasV2(canvas, x, y, flux, section_pos_idx, edgeSersics,n=4, r_eff=10, ellip=0.5, theta=0):
    """
    Add a Sersic profile to the canvas with a check for edges. If the Sersic profile 
    goes out of bounds, save the necessary parameters in edgeSersicArrays for later addition.
    
    Args:
    - canvas: The individual section where the Sersic is being added.
    - x, y: The local position within the section.
    - flux: Total flux of the Sersic profile.
    - section_pos_idx: The corresponding global positions on the stitchedCanvas for the local section.
    - n, r_eff, ellip, theta: Parameters of the Sersic profile.
    - section_idx: The index of the current section.
    """
    canvas_shape = canvas.shape 
    x_min, x_max = x - 5*r_eff, x + 5*r_eff
    y_min, y_max = y - 5*r_eff, y + 5*r_eff

    # Check if the profile goes out of bounds
    if x_min < 0 or x_max >= canvas_shape[0] or y_min < 0 or y_max >= canvas_shape[1]:
        # Use the local (x, y) to find the corresponding global position in section_pos_idx
        global_position_idx = x * canvas_shape[1] + y
        global_x, global_y = section_pos_idx[global_position_idx]

        # Save the necessary parameters in edgeSersicArrays
        edgeSersics.append((global_x, global_y, flux, r_eff, n, ellip, theta))
    else:
        # If within bounds, generate the Sersic profile directly and add to the canvas
        x_grid, y_grid = np.mgrid[:canvas.shape[0], :canvas.shape[1]]
        initial_sersic = Sersic2D(amplitude=1, r_eff=r_eff, n=n, ellip=ellip, theta=theta)
        x_grid_shifted = x_grid - x
        y_grid_shifted = y_grid - y
        initial_profile = initial_sersic(x_grid_shifted, y_grid_shifted)
        amplitude = flux / np.sum(initial_profile)
        sersic = amplitude * initial_profile
        canvas += sersic

def apply_edge_sersics_to_stitched_canvas(stitchedCanvas, edgeSersics):
    """
    Reapply all edge Sersic profiles stored in edgeSersicArrays to the stitchedCanvas.
    
    Args:
    - stitchedCanvas: The large stitched canvas.
    - edgeSersicArrays: List of Sersic parameters to reapply (global_x, global_y, flux, r_eff, n, ellip, theta).
    """ 
    for global_x, global_y, flux, r_eff, n, ellip, theta in tqdm(edgeSersics, desc = 'Edge Sersics Progress:'):
        #generate the Sersic profile
        x_grid, y_grid = np.mgrid[:stitchedCanvas.shape[0], :stitchedCanvas.shape[1]]
        initial_sersic = Sersic2D(amplitude=1, r_eff=r_eff, n=n, ellip=ellip, theta=theta)
        x_grid_shifted = x_grid - global_x
        y_grid_shifted = y_grid - global_y
        initial_profile = initial_sersic(x_grid_shifted, y_grid_shifted)
        
        amplitude = flux / np.sum(initial_profile)
        sersic = amplitude * initial_profile

        # Add the Sersic profile to the stitched canvas
        stitchedCanvas += sersic

    return stitchedCanvas


def process_section(args):
    x, y, flux_column, n_pix, x_range, y_range, dataTable, edgeSersics,section_position_idx = args
    
    x = np.array(x)
    y = np.array(y)
    flux_column = np.array(flux_column)
    
    dataTable = dataTable.copy()  # Make sure the original table isn't modified

    # Now proceed with binned_statistic_2d
    stat, x_edge, y_edge, bin_idx = binned_statistic_2d(
        x, y, values=flux_column, statistic='sum',
        bins=n_pix, range=[x_range, y_range], expand_binnumbers=True
    )

    canvas = np.zeros((n_pix, n_pix))

    for idx, pos in enumerate(bin_idx.T):
        if pos[0] > 0 and pos[0] <= n_pix and pos[1] > 0 and pos[1] <= n_pix:
            #print(f"Adding RA: {dataTable[idx]['ra']}, Dec: {dataTable[idx]['dec']}")
            if dataTable[idx]['type'] == 'PSF' or np.ma.is_masked(dataTable[idx]['type']):  # Check for PSF or masked objects
                canvas[pos[0]-1, pos[1]-1] += flux_column[idx]  # Treat as point source, occupying only one pixel
            else:
                # Use the modified add_sersic_to_canvas_with_edge_check function
                add_sersic_to_canvasV2(canvas, pos[0]-1, pos[1]-1, flux_column[idx], section_position_idx, edgeSersics,
                                                     dataTable[idx]['sersic'], dataTable[idx]['shape_r'], 
                                                     dataTable[idx]['ellipticity'], dataTable[idx]['theta'])
                                                     
    return canvas


def parallel_canvas_generator(ra_column, dec_column, flux_column, ra_offset,dec_offset, dataTable,psf_length, reso_ratio, n_spherex_pix, n_sections):

    start_time = time.time()

    # Project RA/DEC to cartesian coordinate
    rotation_dec = rotation_matrix((90 - dec_offset) * u.deg, axis='x')  # Adjust Declination offset
    rotation_ra = rotation_matrix((ra_offset - 270) * u.deg, axis='z')  # Adjust RA offset from 270
    coords = SkyCoord(ra=ra_column, dec=dec_column, unit='deg')

    # Convert to cartesian coordinate with distance to origin = 1
    cart = coords.represent_as('cartesian')

    rot_cart = cart.transform(rotation_ra @ rotation_dec)    # Apply rotation matrix

    # Get only x and y coordinate, assume flat field approximation to ignore z
    x, y = rot_cart.get_xyz()[0], rot_cart.get_xyz()[1] #x = RA, y = Dec 
    
    # Calculate pixel size and dimensions
    pixel_size = (0.344 * u.arcsec).to(u.degree).value
    # n_spherex_pix = 205 #sqrt
    n_pix = n_spherex_pix * reso_ratio + 2 * psf_length - 1
    angular_d = n_pix * pixel_size
    cart_d = np.sin(angular_d * np.pi / 180)

    stitchedCanvas = np.zeros((n_pix,n_pix))

    # Determine the number of sections per side
    sections_per_side = int(np.sqrt(n_sections))
    section_size = n_pix // sections_per_side

    #print("Obtaining section_positions....")
    section_position_idxs = gen_globalmap_section_indexes(section_size,sections_per_side)
    
    manager = Manager()
    edgeSersics = manager.list()

    ranges = []

    for j in range(sections_per_side):  # Outer loop for rows (y-axis), top to bottom (Declination)
        for i in range(sections_per_side):  # Inner loop for columns (x-axis), left to right (RA)
            x_range = [-cart_d + i * 2 * cart_d / sections_per_side, -cart_d + (i + 1) * 2 * cart_d / sections_per_side]
            y_range = [-cart_d + j * 2 * cart_d / sections_per_side, -cart_d + (j + 1) * 2 * cart_d / sections_per_side]

            #RA and Dec range has been removed!
            
            ranges.append((y_range, x_range)) 

    #List of arguements used for multiprocessing
    args = [(x, y, flux_column, section_size, x_range, y_range, dataTable, edgeSersics, section_position_idxs[idx]) 
            for idx, (x_range, y_range) in enumerate(ranges)]
    
    sectionAreaDeg = (section_size*pixel_size)**2 #area of each section in degrees^2
    totalAreaDeg = (n_pix*pixel_size)**2 #total area of stitchedCanvas in degrees^2

    #print(f'Generating {n_sections} sections using {cpu_count()} cores.\nSection Area: {sectionAreaDeg} deg^2 | Total Area: {totalAreaDeg} deg^2')
    

    # Use multiprocessing to process each section with a progress bar 
    with Pool(cpu_count()) as pool:
        canvas_sections = list(tqdm(pool.imap(process_section, args), total=len(args),desc='TOTAL CANVAS PROGRESS:'))
        
    # Standard recombination of sections into the large canvas
    #stitchedCanvas = np.zeros((sections_per_side * section_size, sections_per_side * section_size))

    #print("Individual Image Processing Complete! Beginning Restitching Process...")
    for canvas_section, global_section_idxs in zip(canvas_sections, section_position_idxs):
        # Flatten the populatedSection to match with section_positions indices
        flat_populated_section = canvas_section.flatten()
        
        # Vectorized assignment: apply the values to stitchedCanvas
        stitchedCanvas[tuple(np.array(global_section_idxs).T)] = flat_populated_section

    #print(f"Restitching Complete! Begin EdgeSersicAdditions (n={len(edgeSersics)})")
    
    stitchedCanvas = apply_edge_sersics_to_stitched_canvas(stitchedCanvas, edgeSersics)

    #no need for this tbh: end_time = time.time()
    elapsed_time = time.time() - start_time
    #print(f'parallel_canvas_generator has been completed. Time Elapsed: {elapsed_time} seconds. Enjoy!')
    return stitchedCanvas

import numpy as np
import cv2  # OpenCV for text overlay
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_section_debug(args):
    """ Generate a debug visualization where each section has a unique color and an index label. """
    section_idx, section_size = args
    
    # Create a section with a unique intensity based on its index
    intensity = int((section_idx / n_sections) * 255)  # Scale from 0-255 for grayscale
    canvas = np.full((section_size, section_size), intensity, dtype=np.uint8)

    # Add a large section index number in the middle
    text = str(section_idx)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Center the text
    text_x = (section_size - text_size[0]) // 2
    text_y = (section_size + text_size[1]) // 2

    # Apply the text onto the section
    cv2.putText(canvas, text, (text_x, text_y), font, font_scale, 255, font_thickness, cv2.LINE_AA)

    return canvas

def debug_stitched_canvas(n_pix, n_sections):
    """ Generates a debug version of stitchedCanvas with indexed section labels. """
    
    # Determine sectioning
    sections_per_side = int(np.sqrt(n_sections))
    section_size = n_pix // sections_per_side
    section_position_idxs = gen_globalmap_section_indexes(section_size, sections_per_side)

    # Generate debug sections
    with Pool(cpu_count()) as pool:
        canvas_sections = list(tqdm(pool.imap(process_section_debug, [(i, section_size) for i in range(n_sections)]), 
                                    total=n_sections, desc='Generating Debug Sections'))

    # Initialize stitched canvas
    stitchedCanvas = np.zeros((n_pix, n_pix), dtype=np.uint8)

    # Stitch the sections together
    for section_idx, (canvas_section, global_section_idxs) in enumerate(zip(canvas_sections, section_position_idxs)):
        # Flatten and apply
        flat_populated_section = canvas_section.flatten()
        stitchedCanvas[tuple(np.array(global_section_idxs).T)] = flat_populated_section

    return stitchedCanvas





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