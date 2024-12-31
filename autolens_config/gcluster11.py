import autolens as al
import autolens.plot as aplt
import numpy as np

def generate_stronger_lensing_galaxy_cluster_with_halo(
    n,
    canvas_size,
    redshift, 
    std_dev,
    einstein_radius_range=(15, 45)
):
    """
    Generates a list of n galaxies with strong lensing properties and includes a dark matter halo.
    
    Parameters:
        n (int): Number of galaxies in the cluster.
        canvas_size (float): Size of the canvas.
        redshift (float): Redshift of the galaxies.
        std_dev (float): Standard deviation of the normal distribution for galaxy positions.
        einstein_radius_range (tuple): Bounds for the 'einstein_radius' in the old isothermal 
                                       code. You can re-purpose these for the new gNFW 
                                       parameters or remove them.
    """
    galaxies = []
    #redshift = np.random.uniform(0.5,5.0)
    
    # Create dark matter halo
    halo_centre = (0.0, 0.0)  # Place at the center of the canvas
    
    # Example usage: We used to randomize the Einstein radius, now we might want to
    # randomize gNFW parameters (kappa_s, scale_radius, inner_slope).
    # For simplicity, here we just sample the "scale_radius" from the old range:
    random_scale_radius = np.random.uniform(*einstein_radius_range)
    
    # Replace Isothermal with SphgNFW (spherical gNFW)
    halo_mass_profile = al.mp.gNFW(
        centre=halo_centre,
        kappa_s=0.2,                # dimensionless amplitude
        scale_radius=random_scale_radius,
        inner_slope=1.5             # typical range ~ [1.0 - 1.5], but can vary
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

        # Random ellipticity and Einstein radius for each galaxy
        ell_comps = (np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4))
        einstein_radius = np.random.uniform(1.5, 3.5)
        
        # Create mass profile for each galaxy (still Isothermal for the main galaxies if you want)
        mass_profile = al.mp.Isothermal(
            centre=(centre_x, centre_y),
            ell_comps=ell_comps,
            einstein_radius=einstein_radius
        )
        
        # Create light profile for each galaxy
        intensity = np.random.uniform(0.6, 1.0)
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
    n_galaxies = np.random.randint(3,8)
    canvas_size = 30.0  # Large canvas size to spread galaxies
    redshift = np.random.uniform(0.5,5)
    std_dev = np.random.uniform(2.0,10.0)      # Standard deviation for the normal distribution of galaxy positions

    # Generate the galaxy cluster with significantly stronger lensing properties
    cluster_galaxies = generate_stronger_lensing_galaxy_cluster_with_halo(
        n=n_galaxies,
        canvas_size=canvas_size,
        redshift =redshift,
        std_dev=std_dev
    )

    # Position the source galaxy close to the cluster for optimal lensing
    source_position = (0, 0)  # You can randomize this as you like
    source_galaxy = al.Galaxy(
        redshift= redshift + np.random.uniform(0.5,1.5),
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
        cluster_plotter = aplt.Array2DPlotter(array=cluster_image)
        cluster_plotter.figure_2d()

        combined_image_plotter = aplt.Array2DPlotter(array=combined_image)
        combined_image_plotter.figure_2d()

    return combined_image.native  # Return native 2D array

if __name__ == "__main__":
    wrapperFunction(verbose=2)

