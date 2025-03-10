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
def add_sersic_to_canvas_flatfield(
    canvas,
    ra_value, dec_value, flux,
    ra_offset, dec_offset,
    psf_length, reso_ratio, n_spherex_pix,
    # Sersic parameters
    n_sersic=4, r_eff=10, ellip=0.5, theta=0
):
    """
    For ONE source at (ra_value, dec_value), do the 'flat field approximation'
    (i.e. rotation and ignoring z), then place a Sersic profile on the 2D canvas.

    We do the snippet of code that used to be in parallel_canvas_generator.
    """

    # ============ FLAT FIELD APPROXIMATION SNIPPET ============
    rotation_dec = rotation_matrix((90 - dec_offset) * u.deg, axis='x')
    rotation_ra  = rotation_matrix((ra_offset - 270) * u.deg, axis='z')
    
    coords = SkyCoord(ra=ra_value, dec=dec_value, unit='deg')
    cart = coords.represent_as('cartesian')
    rot_cart = cart.transform(rotation_ra @ rotation_dec)

    # x,y are the "flattened" coordinates, ignoring z
    x_3d, y_3d, z_3d = rot_cart.get_xyz()
    
    # ============ Decide how to map x_3d,y_3d to pixel coords ============ 
    # For instance, if you want a certain pixel scale:
    pixel_size = (0.344 * u.arcsec).to(u.degree).value  # same as before
    n_pix = n_spherex_pix * reso_ratio + 2 * psf_length - 1
    
    # "cart_d" used to set your bounding box, but you might do:
    #   x_pix = (x_3d - x_3d_min) / (some_scale)
    # or if you're just dropping it into a large local canvas, 
    # you can define a direct offset. For now, let's assume the entire 
    # 2D plane is [-cart_d, +cart_d], so we map x_3d,y_3d => pixel coords:
    #   x_pix = (x_3d + cart_d) / (2*cart_d) * n_pix
    #   y_pix = (y_3d + cart_d) / (2*cart_d) * n_pix
    
    # Example bounding region:
    angular_d = n_pix * pixel_size
    cart_d = np.sin(angular_d * np.pi / 180)

    # Convert 3D -> local pixel coordinate
    x_pix = (x_3d + cart_d) / (2 * cart_d) * n_pix
    y_pix = (y_3d + cart_d) / (2 * cart_d) * n_pix

    # We now have a local (x_pix, y_pix) to place the Sersic
    # ============ Place the Sersic on the 2D canvas ============
    # We can do exactly what your add_sersic_to_canvas does:
    x_grid, y_grid = np.mgrid[:canvas.shape[0], :canvas.shape[1]]
    initial_sersic = Sersic2D(amplitude=1, r_eff=r_eff, n=n_sersic,
                              ellip=ellip, theta=theta)
    
    # Shift grid to center on (x_pix, y_pix)
    x_grid_shifted = x_grid - x_pix
    y_grid_shifted = y_grid - y_pix
    
    initial_profile = initial_sersic(x_grid_shifted, y_grid_shifted)
    
    amplitude = flux / np.sum(initial_profile)
    sersic = amplitude * initial_profile
    
    # Add to canvas
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
    (
        ra_column, dec_column, flux_column,
        n_pix, x_range, y_range, dataTable,
        edgeSersics, section_position_idx,
        ra_offset, dec_offset,
        psf_length, reso_ratio, n_spherex_pix
    ) = args
    
    canvas = np.zeros((n_pix, n_pix))

    # Example loop over each source:
    for i in range(len(ra_column)):
        # If it's a PSF, just plop flux in one pixel or something ...
        if dataTable[i]['type'] == 'PSF':
            ...
        else:
            # For extended objects, do the "flat field" transform 
            # inside add_sersic_to_canvas_flatfield
            flux_val = flux_column[i]
            add_sersic_to_canvas_flatfield(
                canvas,
                ra_column[i],
                dec_column[i],
                flux_val,
                ra_offset,
                dec_offset,
                psf_length,
                reso_ratio,
                n_spherex_pix,
                n_sersic=dataTable[i]['sersic'],
                r_eff=dataTable[i]['shape_r'],
                ellip=dataTable[i]['ellipticity'],
                theta=dataTable[i]['theta']
            )
    return canvas

def parallel_canvas_generator(
    ra_column, dec_column, flux_column, 
    ra_offset, dec_offset, dataTable,
    psf_length, reso_ratio, n_spherex_pix, n_sections
):
    start_time = time.time()

    # (No more rotation or x,y=stuff here—just keep RA, Dec as is.)

    # Set up your final stitchedCanvas
    n_pix = n_spherex_pix * reso_ratio + 2 * psf_length - 1
    stitchedCanvas = np.zeros((n_pix,n_pix))

    sections_per_side = int(np.sqrt(n_sections))
    section_size = n_pix // sections_per_side
    section_position_idxs = gen_globalmap_section_indexes(section_size, sections_per_side)
    
    manager = Manager()
    edgeSersics = manager.list()

    # Still create the sub-ranges, etc. 
    # We'll keep your same logic for dividing the big region if you want.
    # Or you can do something simpler. 
    # Then pass the raw RA/Dec + offsets to each worker.

    args = []
    for idx in range(n_sections):
        # Just a stub example:
        x_range, y_range = ..., ...  # your bounding
        args.append((
            ra_column, dec_column, flux_column,
            section_size, x_range, y_range, dataTable,
            edgeSersics, section_position_idxs[idx],
            ra_offset, dec_offset,  # pass these
            psf_length, reso_ratio, n_spherex_pix
        ))

    with Pool(cpu_count()) as pool:
        canvas_sections = list(
            tqdm(pool.imap(process_section, args), 
                 total=len(args), desc='TOTAL CANVAS PROGRESS:')
        )
    
    # Re-stitch
    for canvas_section, global_section_idxs in zip(canvas_sections, section_position_idxs):
        flattened = canvas_section.flatten()
        stitchedCanvas[tuple(np.array(global_section_idxs).T)] = flattened

    # Edge Sersics re-add:
    stitchedCanvas = apply_edge_sersics_to_stitched_canvas(stitchedCanvas, edgeSersics)
    
    elapsed = time.time() - start_time
    return stitchedCanvas



























#Function for Processing Images to SPHEREx Resolution: 
def processImg(inputImage, psf, psf_length=54, targetResolution=9):
    """
    Processes an image for SPHEREx by convolving it with a PSF and binning the result.
    """
    # Crop and normalize the PSF
    c = int(psf.shape[0] / 2)
    psf = psf[c - psf_length:c + psf_length, c - psf_length:c + psf_length]
    psf = psf / np.sum(psf)

    # Convolve raw image with PSF
    convolved = correlate(inputImage, psf, mode='valid')

    # Bin the image
    def bin2d(img, ratio):
        m_bins = img.shape[0] // ratio
        n_bins = img.shape[1] // ratio
        img_binned = img.reshape(m_bins, ratio, n_bins, ratio).sum(3).sum(1)
        img_binned /= ratio**2
        return img_binned

    binned = bin2d(convolved, targetResolution)

    return convolved, binned





