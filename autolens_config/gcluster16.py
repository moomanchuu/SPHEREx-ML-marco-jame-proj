import autolens as al
import autolens.plot as aplt
import numpy as np

def generate_stronger_lensing_galaxy_cluster_with_halo(
    n,
    canvas_size,
    redshift, 
    std_dev,
    einstein_radius_range=60,
    cluster_offset=(0.0, 0.0)
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
    halo_centre = (cluster_offset[0], cluster_offset[1])  # Place at the center of the canvas

    r_c = n**(0.92)

    
    halo_mass_profile = al.mp.gNFW(
        centre=halo_centre,
        kappa_s=0.05,                # dimensionless amplitude
        scale_radius=r_c*25*std_dev,
        inner_slope=1.5             # typical range ~ [1.0 - 1.5], but can vary
    )
    
    # Add the dark matter halo as an invisible galaxy
    dark_matter_halo = al.Galaxy(redshift=redshift, mass=halo_mass_profile)
    galaxies.append(dark_matter_halo)
    
    for _ in range(n):
        centre_x = np.random.normal(loc=0.0, scale=9*std_dev) + cluster_offset[0]
        centre_y = np.random.normal(loc=0.0, scale=9*std_dev) + cluster_offset[1]

        r = np.sqrt((centre_x**2) + (centre_y**2))

        # Ensure galaxies are within the canvas bounds
        # while abs(centre_x) > canvas_size / 2 or abs(centre_y) > canvas_size / 2:
        #     centre_x = np.random.normal(loc=0.0, scale=std_dev)
        #     centre_y = np.random.normal(loc=0.0, scale=std_dev)

        # Random ellipticity and Einstein radius for each galaxy
        ell_comps = (np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4))
        einstein_radius = np.random.uniform(3.0, 6.0)
        
        # Create mass profile for each galaxy (still Isothermal for the main galaxies if you want)
        mass_profile = al.mp.Isothermal(
            centre=(centre_x, centre_y),
            ell_comps=ell_comps,
            einstein_radius=einstein_radius
        )

        # mass_profile = al.mp.Sersic(
        #     centre=(centre_x, centre_y),
        #     ell_comps=ell_comps,
        #     intensity=1.0  # Lower the intensity
        # )
        
        # Create light profile for each galaxy
        # base_intense = np.random.uniform(3.0,15.0)
        # intense = base_intense*(1/(1+r/r_c)) # higher for closer to center
        # light_profile = al.lp.Sersic(
        #     centre=halo_centre,
        #     ell_comps=ell_comps,
        #     intensity=intense,
        #     effective_radius=intense*1.5,
        #     sersic_index=np.random.uniform(1.0,4.0)
        # )
        intensity = np.random.uniform(0.5, 3.5)
        # effective_radius = np.random.uniform(0.3, 0.6)
        light_profile = al.lp.Sersic(
            centre=(centre_x, centre_y),
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=intensity*0.6, 
            sersic_index=np.random.uniform(1.0,4.0)
        )

        # Add galaxy to the list
        galaxy = al.Galaxy(
            redshift=redshift,
            mass=mass_profile,
            light=light_profile
        )
        galaxies.append(galaxy)
    
    return galaxies

def wrapperFunction(n_galaxies, background_image, multiple, verbose=2):
    if multiple == True:
        n_clusters=2
        canvas_size = 100.0    # Define overall canvas size.
        redshift = 0.5
        std_dev = np.random.uniform(3.0, 10.0)
        
        # Generate random offsets for each cluster. Here we choose offsets within ±canvas_size/3.
        cluster_offsets = []
        offset_limit = canvas_size*2
        for _ in range(n_clusters):
            offset_x = np.random.uniform(-offset_limit, offset_limit)
            offset_y = np.random.uniform(-offset_limit, offset_limit)
            cluster_offsets.append((offset_x, offset_y))
        
        # Generate the galaxies for all clusters and combine them.

        # For now, lets randomize the number of galaxies
        all_clusters_galaxies = []
        for offset in cluster_offsets:
            N = int(np.random.uniform(7,10))
            cluster_galaxies = generate_stronger_lensing_galaxy_cluster_with_halo(
                n=N,
                canvas_size=canvas_size,
                redshift=redshift,
                std_dev=std_dev,
                cluster_offset=offset
            )
            all_clusters_galaxies += cluster_galaxies
        
        # Define a source galaxy placed at the center.
        source_position = (0.0, 0.0)
        intensity = np.random.uniform(0.5, 1.0)
        source_profile = al.lp.Sersic(
            centre=source_position,
            intensity=intensity,
            effective_radius=intensity * 1.5, 
            sersic_index=np.random.uniform(1.0, 4.0)
        )
        source_galaxy = al.Galaxy(redshift=1.5, light=source_profile)
        
        if background_image.any() == None:
            # Create the simulation grid.
            grid = al.Grid2D.uniform(
                shape_native=(600, 600),
                pixel_scales=1.8
            )
            
            # First, produce the image from all cluster galaxies (their own light).
            tracer_clusters_only = al.Tracer(galaxies=all_clusters_galaxies)
            clusters_image = tracer_clusters_only.image_2d_from(grid=grid)
            
            # Now produce the full lensed image (clusters + source).
            tracer_with_source = al.Tracer(galaxies=all_clusters_galaxies + [source_galaxy])
            lensed_image_with_clusters = tracer_with_source.image_2d_from(grid=grid)
            
            # Combine the cluster light and the lensed source image.
            combined_image = lensed_image_with_clusters + clusters_image
            
            return combined_image.native

        else:
            grid = al.Grid2D.uniform(
                shape_native=background_image.shape,  # Match the background image dimensions
                pixel_scales=3.1,  # Set the pixel scale
            )

            tracer_clusters_only = al.Tracer(galaxies=all_clusters_galaxies)
            clusters_image = tracer_clusters_only.image_2d_from(grid=grid)
            
            # Now produce the full lensed image (clusters + source).
            tracer_with_source = al.Tracer(galaxies=all_clusters_galaxies + [source_galaxy])
            lensed_image_with_clusters = tracer_with_source.image_2d_from(grid=grid)
            
            # Combine the cluster light and the lensed source image.
            combined_image = background_image + lensed_image_with_clusters.native + clusters_image.native

            return combined_image

    else:
        # Increase the number of galaxies in the cluster and canvas size
        # n_galaxies = np.random.randint(3,8)
        canvas_size = 100.0  # Large canvas size to spread galaxies
        redshift = 0.5
        std_dev = np.random.uniform(3.0,10.0)   # Standard deviation for the normal distribution of galaxy positions

        # Generate the galaxy cluster with significantly stronger lensing properties
        cluster_galaxies = generate_stronger_lensing_galaxy_cluster_with_halo(
            n=n_galaxies,
            canvas_size=canvas_size,
            redshift =redshift,
            std_dev=std_dev
        )

        # Position the source galaxy close to the cluster for optimal lensing
        source_position = (
            np.random.uniform(0.0,5.0),
            np.random.uniform(0.0,5.0)
        )  # You can randomize this as you like
        intensity = np.random.uniform(0.5, 1.0)
        effective_radius = np.random.uniform(0.3, 0.6)
        source_profile = al.lp.Sersic(
            centre=source_position,
            intensity=intensity,
            effective_radius=intensity*1.5, 
            sersic_index=np.random.uniform(1.0,4.0)
        )

        source_galaxy = al.Galaxy(
            redshift=1.5,
            light = source_profile
        )

        if background_image.any() == None: # .any()
        # Define a grid for the simulation to cover the entire canvas
            grid = al.Grid2D.uniform(
                shape_native=(600, 600),
                pixel_scales=1.8  # Pixel scale for capturing fine details
            )

            # Create a tracer for just the galaxy cluster (lens galaxies only, without the source)
            tracer_cluster_only = al.Tracer(galaxies=cluster_galaxies)

            # Generate the image of the galaxy cluster alone
            cluster_image = tracer_cluster_only.image_2d_from(grid=grid)

            # critical_curve_grid = compute_critical_curve(tracer_cluster_only, grid)


            # Create a tracer including both the galaxy cluster and the lensed source galaxy
            tracer_with_source = al.Tracer(galaxies=cluster_galaxies + [source_galaxy])

            # Generate the full lensed image including both the galaxy cluster and the lensed source galaxy
            lensed_image_with_cluster = tracer_with_source.image_2d_from(grid=grid)

            # Combine the two images by adding the cluster light to the lensed image
            combined_image = lensed_image_with_cluster + cluster_image

            # if verbose >= 2:
            #     cluster_plotter = aplt.Array2DPlotter(array=cluster_image)
            #     cluster_plotter.figure_2d()

                # combined_image_plotter = aplt.Array2DPlotter(array=combined_image)
                # combined_image_plotter.figure_2d()

            return combined_image.native  # Return native 2D array
            # return cluster_image.native
        
        else:
            grid = al.Grid2D.uniform(
                shape_native=background_image.shape,  # Match the background image dimensions
                pixel_scales=3.1,  # Set the pixel scale
            )
            tracer_cluster_only = al.Tracer(galaxies=cluster_galaxies)
            cluster_image = tracer_cluster_only.image_2d_from(grid=grid)

            tracer_with_source = al.Tracer(galaxies=cluster_galaxies + [source_galaxy])
            lensed_image_with_cluster = tracer_with_source.image_2d_from(grid=grid)


            combined_image = background_image + lensed_image_with_cluster.native + cluster_image.native 

            return combined_image
 

if __name__ == "__main__":
    wrapperFunction(verbose=2)

