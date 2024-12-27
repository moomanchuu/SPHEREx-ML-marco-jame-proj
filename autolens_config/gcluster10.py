import autolens as al
import autolens.plot as aplt
import numpy as np



def generate_stronger_lensing_galaxy_cluster_with_halo(
    n, canvas_size=30.0, redshift=0.5, std_dev=5.0, einstein_radius_range=(15, 45)
):
    """
    Generates a list of n galaxies with strong lensing properties and includes a dark matter halo.
    
    Parameters:
        n (int): Number of galaxies in the cluster.
        canvas_size (float): Size of the canvas.
        redshift (float): Redshift of the galaxies.
        std_dev (float): Standard deviation of the normal distribution for galaxy positions.
        einstein_radius_range (tuple): Bounds for the Einstein radius of the dark matter halo in arcseconds.
    """
    galaxies = []
    
    # Create dark matter halo
    halo_centre = (0.0, 0.0)  # Place at the center of the canvas
    halo_einstein_radius = np.random.uniform(*einstein_radius_range)
    halo_mass_profile = al.mp.Isothermal(
        centre=halo_centre,
        ell_comps=(0.0, 0.0),  # No ellipticity for simplicity
        einstein_radius=halo_einstein_radius
    )
    
    # Add the dark matter halo as an invisible galaxy
    dark_matter_halo = al.Galaxy(redshift=redshift, mass=halo_mass_profile)
    galaxies.append(dark_matter_halo)
    
    for _ in range(n):
        centre_x = np.random.normal(loc=0.0, scale=std_dev)
        centre_y = np.random.normal(loc=0.0, scale=std_dev)

        # Ensure galaxies are within the canvas bounds
        while abs(centre_x) > canvas_size / 2 or abs(centre_y) > canvas_size / 2:
            centre_x = np.random.normal(loc=0.0, scale=std_dev)
            centre_y = np.random.normal(loc=0.0, scale=std_dev)

        # Increase ellipticity and Einstein radius for individual galaxies
        ell_comps = (np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4))
        einstein_radius = np.random.uniform(1.5, 2.5)
        
        # Create mass profile for galaxy
        mass_profile = al.mp.Isothermal(
            centre=(centre_x, centre_y),
            ell_comps=ell_comps,
            einstein_radius=einstein_radius
        )
        
        # Create light profile for galaxy
        intensity = np.random.uniform(0.4, 0.8)
        effective_radius = np.random.uniform(0.3, 0.6)
        light_profile = al.lp.ExponentialSph(
            centre=(centre_x, centre_y),
            intensity=intensity,
            effective_radius=effective_radius
        )

        # Add galaxy to the list
        galaxy = al.Galaxy(
            redshift=redshift,
            mass=mass_profile,
            light=light_profile
        )
        galaxies.append(galaxy)
    
    return galaxies

def wrapperFunction(verbose=2):
    # Increase the number of galaxies in the cluster and canvas size
    n_galaxies = 10
    canvas_size = 30.0  # Large canvas size to spread galaxies
    std_dev = 5.0  # Standard deviation for the normal distribution of galaxy positions

    # Generate the galaxy cluster with significantly stronger lensing properties
    cluster_galaxies = generate_stronger_lensing_galaxy_cluster_with_halo(
        n=n_galaxies,
        canvas_size=canvas_size,
        std_dev=std_dev
    )

    # Position the source galaxy close to the cluster for optimal lensing
    source_x = np.random.uniform(-canvas_size/3, canvas_size/3)
    source_y = np.random.uniform(-canvas_size/3, canvas_size/3)
    source_position = (0, 0)  # Offset source position for optimal arcs

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.ExponentialSph(
            centre=source_position,
            intensity=0.5,
            effective_radius=0.3
        )
    )

    # Define a grid for the simulation to cover the entire canvas
    grid = al.Grid2D.uniform(
        shape_native=(395, 395),
        pixel_scales=0.20  # Pixel scale for capturing fine details
    )

    # Create a tracer for just the galaxy cluster (lens galaxies only, without the source)
    tracer_cluster_only = al.Tracer(galaxies=cluster_galaxies)

    # Generate the image of the galaxy cluster alone
    cluster_image = tracer_cluster_only.image_2d_from(grid=grid)

    # Create a tracer including both the galaxy cluster and the lensed source galaxy
    tracer_with_source = al.Tracer(galaxies=cluster_galaxies + [source_galaxy])

    # Generate the full lensed image including both the galaxy cluster and the lensed source galaxy
    lensed_image_with_cluster = tracer_with_source.image_2d_from(grid=grid)

    # Combine the two images by adding the cluster light to the lensed image
    combined_image = lensed_image_with_cluster + cluster_image

    if verbose >= 2:

        # Plot the light from the galaxy cluster alone
        cluster_plotter = aplt.Array2DPlotter(array=cluster_image)
        cluster_plotter.figure_2d()

        combined_image_plotter = aplt.Array2DPlotter(array=combined_image)
        combined_image_plotter.figure_2d()

    return combined_image.native #want it as native since then I can use numpy 2D stuff


if __name__ == "__main__":
    # Adjust verbosity or parameters here if needed
    wrapperFunction(verbose=2)




''' #Random stuff I gpt'd, prob delete lol
def create_galaxy_cluster(n_galaxies, canvas_size, std_dev):
    """
    Creates a galaxy cluster with a specified number of galaxies and random properties.

    Parameters:
        n_galaxies (int): Number of galaxies in the cluster.
        canvas_size (float): Size of the canvas for galaxy positions.
        std_dev (float): Standard deviation for galaxy position distribution.

    Returns:
        list: A list of galaxies in the cluster.
    """
    galaxies = []
    for _ in range(n_galaxies):
        # Random positions from a normal distribution centered at 0
        centre_x = np.random.normal(loc=0.0, scale=std_dev)
        centre_y = np.random.normal(loc=0.0, scale=std_dev)

        # Ensure galaxies are within the canvas bounds
        while abs(centre_x) > canvas_size / 2 or abs(centre_y) > canvas_size / 2:
            centre_x = np.random.normal(loc=0.0, scale=std_dev)
            centre_y = np.random.normal(loc=0.0, scale=std_dev)

        # Random ellipticity and Einstein radius for lensing strength
        ell_comps = (np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4))
        einstein_radius = np.random.uniform(1.5, 2.5)

        # Create a mass profile
        mass_profile = al.mp.Isothermal(
            centre=(centre_x, centre_y),
            ell_comps=ell_comps,
            einstein_radius=einstein_radius
        )

        # Create a light profile for appearance
        intensity = np.random.uniform(0.4, 0.8)
        effective_radius = np.random.uniform(0.3, 0.6)

        # Create a galaxy with mass and light profiles
        galaxy = al.Galaxy(
            redshift=0.5,
            mass=mass_profile,
            light=al.lp.ExponentialSph(
                centre=(centre_x, centre_y),
                intensity=intensity,
                effective_radius=effective_radius
            )
        )
        galaxies.append(galaxy)

    return galaxies


def create_lensed_object(source_redshift, source_position, source_intensity, source_effective_radius, dm_centre, dm_einstein_radius):
    """
    Creates the lensed object, including the source galaxy and dark matter halo.

    Parameters:
        source_redshift (float): Redshift of the source galaxy.
        source_position (tuple): Position of the source galaxy (x, y).
        source_intensity (float): Intensity of the source galaxy's light profile.
        source_effective_radius (float): Effective radius of the source galaxy's light profile.
        dm_centre (tuple): Centre of the dark matter halo (x, y).
        dm_einstein_radius (float): Einstein radius of the dark matter halo (in arcseconds).

    Returns:
        tuple: A tuple containing the source galaxy and dark matter halo as Galaxy objects.
    """
    # Create the source galaxy
    source_galaxy = al.Galaxy(
        redshift=source_redshift,
        light=al.lp.ExponentialSph(
            centre=source_position,
            intensity=source_intensity,
            effective_radius=source_effective_radius
        )
    )

    # Create the dark matter halo
    dm_halo = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=dm_centre,
            ell_comps=(0.0, 0.0),  # Assume spherical halo
            einstein_radius=dm_einstein_radius
        )
    )

    return source_galaxy, dm_halo


def simulate_lensed_galaxy_cluster(
    n_galaxies=None,
    canvas_size=None,
    std_dev=None,
    grid_shape=None,
    pixel_scales=None,
    source_redshift=None,
    source_position=None,
    source_intensity=None,
    source_effective_radius=None,
    dm_centre=None,
    dm_einstein_radius=None,
    verbose=0
):
    """
    Wrapper function that uses the galaxy cluster and lensed object creation functions to simulate
    a full lensed galaxy cluster.

    Parameters:
        All parameters are optional and will be randomly generated if not provided.
        verbose (int): 
            0 - No output.
            1 - Print basic status messages.
            2 - Display the generated image.

    Returns:
        combined_image (Array2D): The combined image of the galaxy cluster and lensed object.
    """
    # Set defaults or generate random parameters
    n_galaxies = n_galaxies or 10
    canvas_size = canvas_size or 30.0 
    std_dev = std_dev or 5.0 #Standard deviation for the normal distribution of galaxy positions
    grid_shape = grid_shape or (395, 395) #See SPHEREx Image Size Requirements...
    pixel_scales = pixel_scales or 0.20 #Similar in pixel resolution to Subaru Telescope
    source_redshift = source_redshift or 1.0 
    source_position = source_position or (np.random.uniform(-canvas_size / 3, canvas_size / 3),
                                          np.random.uniform(-canvas_size / 3, canvas_size / 3))
    source_intensity = source_intensity or np.random.uniform(0.4, 0.8)
    source_effective_radius = source_effective_radius or np.random.uniform(0.3, 0.6)
    dm_centre = dm_centre or (0.0, 0.0)
    dm_einstein_radius = dm_einstein_radius or np.random.uniform(15.0, 45.0)

    if verbose >= 1:
        print(f"Generating galaxy cluster with {n_galaxies} galaxies.")
        print(f"Creating lensed object with DM Einstein radius: {dm_einstein_radius} arcseconds.")

    #Generate galaxy cluster: 
    cluster_galaxies = create_galaxy_cluster(n_galaxies, canvas_size, std_dev)


    #Generate lensed object
    source_galaxy, dm_halo = create_lensed_object(
        source_redshift,
        source_position,
        source_intensity,
        source_effective_radius,
        dm_centre,
        dm_einstein_radius
    )

    # Define the simulation grid
    grid = al.Grid2D.uniform(
        shape_native=grid_shape,
        pixel_scales=pixel_scales
    )

    # Create tracers
    tracer_with_source = al.Tracer(galaxies=cluster_galaxies + [dm_halo, source_galaxy])
    lensed_image_with_cluster = tracer_with_source.image_2d_from(grid=grid)

    tracer_cluster_only = al.Tracer(galaxies=cluster_galaxies + [dm_halo])
    cluster_image = tracer_cluster_only.image_2d_from(grid=grid)

    # Combine images
    combined_image = lensed_image_with_cluster + cluster_image

    if verbose >= 2:
        combined_image_plotter = aplt.Array2DPlotter(array=combined_image)
        combined_image_plotter.figure_2d()

    return combined_image


if __name__ == "__main__":
    # Adjust verbosity or parameters here if needed
    simulate_lensed_galaxy_cluster(verbose=2)
'''