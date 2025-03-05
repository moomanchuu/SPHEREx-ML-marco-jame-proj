import autolens as al
import autolens.plot as aplt
import numpy as np

def generate_stronger_lensing_galaxy_cluster_with_halo(
    n,
    canvas_size,
    redshift, 
    std_dev,
    einstein_radius_range=60,
    cluster_offset=(0.0, 0.0),
    is_single = False
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
    
    # Create dark matter halo
    halo_centre = (cluster_offset[0], cluster_offset[1])  # Place at the center of the canvas

    r_c = n**(0.92)

    if is_single == True:
        kap = 0.009
        einstein_radius = np.random.uniform(4.0,6.0)
        scale_fac = 60*std_dev
        cent_scal = 10
        off_scal = 1.2
    else:
        kap = 0.04
        einstein_radius = np.random.uniform(5.0,7.0)
        scale_fac = 50*std_dev
        cent_scal = 9
        off_scal = 1

    # make the dark matter halo more elliptical?
    dark_comp = 0.25
    dark_ell = (np.random.uniform(-dark_comp, dark_comp), np.random.uniform(-dark_comp, dark_comp))

    
    halo_mass_profile = al.mp.gNFW(
        centre=halo_centre,
        ell_comps=dark_ell,
        kappa_s=kap,           # used to be 0.008     # dimensionless amplitude
        scale_radius=r_c*scale_fac, # 50 before
        inner_slope= 1.85             # typical range ~ [1.0 - 1.5], but can vary
    )
    
    # Add the dark matter halo as an invisible galaxy
    dark_matter_halo = al.Galaxy(redshift=redshift, mass=halo_mass_profile)
    galaxies.append(dark_matter_halo)
    
    for _ in range(n):
        centre_x = np.random.normal(loc=0.0, scale=cent_scal*std_dev) + off_scal*cluster_offset[0] # scale = 9* std_dev before, 1*cluster offset
        centre_y = np.random.normal(loc=0.0, scale=cent_scal*std_dev) + off_scal*cluster_offset[1]

        r = np.sqrt((centre_x**2) + (centre_y**2))

        # Ensure galaxies are within the canvas bounds
        # while abs(centre_x) > canvas_size / 2 or abs(centre_y) > canvas_size / 2:
        #     centre_x = np.random.normal(loc=0.0, scale=std_dev)
        #     centre_y = np.random.normal(loc=0.0, scale=std_dev)

        # Random ellipticity and Einstein radius for each galaxy
        ell_num = 0.5
        ell_comps = (np.random.uniform(-ell_num, ell_num), np.random.uniform(-ell_num, ell_num))
        # einstein_radius = np.random.uniform(3.0, 7.0)
        
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

def wrapperFunction(n_galaxies, background_image, multiple, canvas_size, verbose=2):
    if multiple == True:
        n_clusters=3
        # canvas_size = 200.0    # Define overall canvas size.
        redshift = 0.5
        std_dev = np.random.uniform(3.0, 10.0)
        
        # Generate random offsets for each cluster. Here we choose offsets within 
        cluster_offsets = []
        offset_limit = 395 # canvas_size*2.0
        for _ in range(n_clusters):
            offset_x = np.random.uniform(-offset_limit, offset_limit)
            offset_y = np.random.uniform(-offset_limit, offset_limit)
            cluster_offsets.append((offset_x, offset_y))
        
        # Generate the galaxies for all clusters and combine them.

        # For now, lets randomize the number of galaxies
        all_clusters_galaxies = []
        for offset in cluster_offsets:
            # N = int(np.random.uniform(n_galaxies-2,n_galaxies+2))
            N = n_galaxies
            cluster_galaxies = generate_stronger_lensing_galaxy_cluster_with_halo(
                n=N,
                canvas_size=canvas_size,
                redshift=redshift,
                std_dev=std_dev,
                cluster_offset=offset
            )
            all_clusters_galaxies += cluster_galaxies
        
        # # Define a source galaxy placed at the center.
        # source_position = (0.0, 0.0)
        # intensity = np.random.uniform(0.5, 1.0)
        # source_profile = al.lp.Sersic(
        #     centre=source_position,
        #     intensity=intensity,
        #     effective_radius=intensity * 1.5, 
        #     sersic_index=np.random.uniform(1.0, 4.0)
        # )
        # source_galaxy = al.Galaxy(redshift=1.5, light=source_profile)
        
        if background_image.any() == None:
            # See Multiple-None at bottom of file. 
            print("No background img code in function ")
            return 0

        else:
            # grid = al.Grid2D.uniform(
            #     shape_native=background_image.shape,  # Match the background image dimensions
            #     pixel_scales=3.1,  # Set the pixel scale
            # )

            # Convert to NumPy array for easier computation
            offsets_array = np.array(cluster_offsets)

            # Compute the centroid (mean of x and y values)
            center_x = np.mean(offsets_array[:, 0])
            center_y = np.mean(offsets_array[:, 1])

            grid_sub = al.Grid2D.uniform(
                shape_native= (395,395),
                pixel_scales= 3.1,
                origin=(center_x, center_y)  # centers the subgrid on the cluster. For multiple clusters take the center still?
            )

            tracer_clusters_only = al.Tracer(galaxies=all_clusters_galaxies)
            clusters_image = tracer_clusters_only.image_2d_from(grid=grid_sub)

            tangential_caustics = tracer_clusters_only.tangential_caustic_list_from(grid=grid_sub)

            # Extract tangential caustic points
            tangential_caustic_points = tangential_caustics[0]

            # Find the cusp with the highest local curvature
            # A simple approach: select points where the x or y coordinate changes most sharply
            diffs = np.abs(np.diff(tangential_caustic_points, axis=0))
            curvature_index = np.argmax(diffs[:, 0] + diffs[:, 1])  # Select highest curvature point

            # Place source near this cusp
            max_mag = tangential_caustic_points[curvature_index]


            # print(f"Maximum magnification occurs at {max_mag}")
            # print(max_mag[0], max_mag[1])

            # # Define a source galaxy placed at the highest magnification
            source_position = (max_mag[0], max_mag[1])
            intensity = np.random.uniform(0.5, 1.0)
            source_profile = al.lp.Sersic(
                centre=source_position,
                intensity=intensity,
                effective_radius=intensity * 1.5, 
                sersic_index=np.random.uniform(1.0, 4.0)
            )
            source_galaxy = al.Galaxy(redshift=1.5, light=source_profile)
            
            # Now produce the full lensed image (clusters + source).
            tracer_with_source = al.Tracer(galaxies=all_clusters_galaxies + [source_galaxy])
            lensed_image_with_clusters = tracer_with_source.image_2d_from(grid=grid_sub)


            H_bg, W_bg = background_image.shape 

            stamp_size = 395
            # Random offsets that ensure the stamp fits entirely:
            y_offset = np.random.randint(0, H_bg - stamp_size + 1)
            x_offset = np.random.randint(0, W_bg - stamp_size + 1)

            background_image[y_offset:y_offset+stamp_size, x_offset:x_offset+stamp_size] += lensed_image_with_clusters.native
            
            # Combine the cluster light and the lensed source image.
            # combined_image = background_image + lensed_image_with_clusters.native + clusters_image.native

            return background_image

    else:
        # Increase the number of galaxies in the cluster and canvas size
        # n_galaxies = np.random.randint(3,8)
        is_single = True

        # canvas_size = 200.0  # Large canvas size to spread galaxies
        redshift = 0.5
        std_dev = np.random.uniform(3.0,10.0)   # Standard deviation for the normal distribution of galaxy positions

        cluster_offset = []
        offset_limit = 395 # canvas_size # Originally size*2

        offset_x = np.random.uniform(-offset_limit, offset_limit)
        offset_y = np.random.uniform(-offset_limit, offset_limit)
        cluster_offset.append((offset_x, offset_y))

        # Generate the galaxy cluster with significantly stronger lensing properties
        cluster_galaxies = generate_stronger_lensing_galaxy_cluster_with_halo(
            n=n_galaxies,
            canvas_size=canvas_size,
            redshift =redshift,
            std_dev=std_dev,
            cluster_offset=cluster_offset[0],
            is_single=is_single
        )

        if background_image.any() == None: # .any()
            # See Single-None at bottom of file
            print("No background img code in function ")
            return 0
        
        else:
            # grid = al.Grid2D.uniform(
            #     shape_native=background_image.shape,  # Match the background image dimensions
            #     pixel_scales=3.1,  # Set the pixel scale, background is 0.344
            # )

            grid_sub = al.Grid2D.uniform(
                shape_native= (395,395),
                pixel_scales= 3.1,
                origin=cluster_offset[0]  # centers the subgrid on the cluster
            )

            tracer_cluster_only = al.Tracer(galaxies=cluster_galaxies)
            cluster_image = tracer_cluster_only.image_2d_from(grid=grid_sub)


            tangential_caustics = tracer_cluster_only.tangential_caustic_list_from(grid=grid_sub)

            # Extract tangential caustic points
            tangential_caustic_points = tangential_caustics[0]

            # Find the cusp with the highest local curvature
            # A simple approach: select points where the x or y coordinate changes most sharply
            diffs = np.abs(np.diff(tangential_caustic_points, axis=0))
            curvature_index = np.argmax(diffs[:, 0] + diffs[:, 1])  # Select highest curvature point

            # Place source near this cusp
            max_mag = tangential_caustic_points[curvature_index]


            # print(f"Maximum magnification occurs at {max_mag}")
            # print(max_mag[0], max_mag[1])

            # # Define a source galaxy placed at the highest magnification
            # source_position = (max_mag[0] + cluster_offset[0][0], max_mag[1] + cluster_offset[0][1])
            source_position = (max_mag[0], max_mag[1])
            intensity = np.random.uniform(0.5, 1.0)
            source_profile = al.lp.Sersic(
                centre=source_position,
                intensity=intensity,
                effective_radius=intensity * 1.5, 
                sersic_index=np.random.uniform(1.0, 4.0)
            )
            source_galaxy = al.Galaxy(redshift=1.5, light=source_profile)

            tracer_with_source = al.Tracer(galaxies=cluster_galaxies + [source_galaxy])
            # lensed_image_with_cluster = tracer_with_source.image_2d_from(grid=grid)
            lensed_image_with_cluster = tracer_with_source.image_2d_from(grid=grid_sub)

            H_bg, W_bg = background_image.shape 

            stamp_size = 395
            # Random offsets that ensure the stamp fits entirely:
            y_offset = np.random.randint(0, H_bg - stamp_size + 1)
            x_offset = np.random.randint(0, W_bg - stamp_size + 1)

            background_image[y_offset:y_offset+stamp_size, x_offset:x_offset+stamp_size] += lensed_image_with_cluster.native

            # combined_image = background_image[y_offset:y_offset+stamp_size, x_offset:x_offset+stamp_size] + lensed_image_with_cluster.native


            # combined_image = background_image + lensed_image_with_cluster.native#  + cluster_image.native 

            # array_plotter = aplt.Array2DPlotter(
            #     array=combined_image, # change to lensed_image -> None for source plane
            #     visuals_2d=visuals,
            #     mat_plot_2d=mat_plot
            # )
            # array_plotter.figure_2d()

            # print(cluster_offset[0])

            return background_image #, combined_image
            # return combined_image
 

if __name__ == "__main__":
    wrapperFunction(verbose=2)

# Multiple-None
# -------------
# # Create the simulation grid.
# grid = al.Grid2D.uniform(
#     shape_native=(600, 600),
#     pixel_scales=0.8
# )

# # First, produce the image from all cluster galaxies (their own light).
# tracer_clusters_only = al.Tracer(galaxies=all_clusters_galaxies)
# clusters_image = tracer_clusters_only.image_2d_from(grid=grid)


# # Compute caustics
# radial_caustics = tracer_clusters_only.radial_caustic_list_from(grid=grid)
# tangential_caustics = tracer_clusters_only.tangential_caustic_list_from(grid=grid)

# # Extract tangential caustic points
# tangential_caustic_points = tangential_caustics[0]

# # Find the cusp with the highest local curvature
# # A simple approach: select points where the x or y coordinate changes most sharply
# diffs = np.abs(np.diff(tangential_caustic_points, axis=0))
# curvature_index = np.argmax(diffs[:, 0] + diffs[:, 1])  # Select highest curvature point

# # Place source near this cusp
# max_mag = tangential_caustic_points[curvature_index]


# print(f"Maximum magnification occurs at {max_mag}")
# # print(max_mag[0], max_mag[1])

# # # Define a source galaxy placed at the highest magnification
# source_position = (max_mag[0], max_mag[1])
# intensity = np.random.uniform(0.5, 1.0)
# source_profile = al.lp.Sersic(
#     centre=source_position,
#     intensity=intensity,
#     effective_radius=intensity * 1.5, 
#     sersic_index=np.random.uniform(1.0, 4.0)
# )
# source_galaxy = al.Galaxy(redshift=1.5, light=source_profile)


# # Now produce the full lensed image (clusters + source).
# tracer_with_source = al.Tracer(galaxies=all_clusters_galaxies + [source_galaxy])
# lensed_image_with_clusters = tracer_with_source.image_2d_from(grid=grid)

# # Combine the cluster light and the lensed source image.
# combined_image = lensed_image_with_clusters + clusters_image

# # Compute lensed image
# lensed_image = tracer_clusters_only.image_2d_from(grid=grid)

# # Define visualization settings
# visuals = aplt.Visuals2D(
#     radial_caustics=radial_caustics,
#     tangential_caustics=tangential_caustics
# )

# mat_plot = aplt.MatPlot2D(
#     output=aplt.Output(
#         path="./output",
#         filename="caustics_cluster_lensing",
#         format="png",
#         bbox_inches="tight"
#     )
# )

# # Plot the result
# array_plotter = aplt.Array2DPlotter(
#     array=lensed_image,
#     visuals_2d=visuals,
#     mat_plot_2d=mat_plot
# )
# array_plotter.figure_2d()
# 
# return combined_image.native


# Single-None
# -----------
# Define a grid for the simulation to cover the entire canvas
# grid = al.Grid2D.uniform(
#     shape_native=(600, 600),
#     pixel_scales=0.6  # 1.8 before testing for critical curves # Pixel scale for capturing fine details
# )

# # Create a tracer for just the galaxy cluster (lens galaxies only, without the source)
# tracer_cluster_only = al.Tracer(galaxies=cluster_galaxies)

# # Generate the image of the galaxy cluster alone
# cluster_image = tracer_cluster_only.image_2d_from(grid=grid)

# # critical_curve_grid = compute_critical_curve(tracer_cluster_only, grid)


# # Create a tracer including both the galaxy cluster and the lensed source galaxy
# tracer_with_source = al.Tracer(galaxies=cluster_galaxies + [source_galaxy])

# # Generate the full lensed image including both the galaxy cluster and the lensed source galaxy
# lensed_image_with_cluster = tracer_with_source.image_2d_from(grid=grid)

# # Combine the two images by adding the cluster light to the lensed image
# combined_image = lensed_image_with_cluster + cluster_image

# # Compute caustics
# # radial_caustics = tracer_with_source.radial_caustic_list_from(grid=grid)
# tangential_caustics = tracer_with_source.tangential_caustic_list_from(grid=grid)

# # Compute lensed image
# # lensed_image = tracer_with_source.image_2d_from(grid=grid)

# # Define visualization settings
# visuals = aplt.Visuals2D(
#     # radial_caustics=radial_caustics,
#     tangential_caustics=tangential_caustics
# )

# mat_plot = aplt.MatPlot2D(
#     output=aplt.Output(
#         path="./output",
#         filename="caustics_cluster_lensing",
#         format="png",
#         bbox_inches="tight"
#     )
# )

# # empty_array = al.Array2D(
# #     array=np.zeros((600, 600)), 
# #     pixel_scales=0.4
# # )

# # Plot the result
# array_plotter = aplt.Array2DPlotter(
#     array=combined_image, 
#     visuals_2d=visuals,
#     mat_plot_2d=mat_plot
# )
# array_plotter.figure_2d()
# return combined_image.native  # Return native 2D array
# # return cluster_image.native