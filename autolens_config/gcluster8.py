import autolens as al
import autolens.plot as aplt
import numpy as np

def generate_stronger_lensing_galaxy_cluster(n, canvas_size, redshift=0.5):
    """Generates a list of n galaxies with very strong lensing properties randomly distributed across the canvas."""
    galaxies = []
    pos_scale = 1.5
    for _ in range(n):
        # Random positions across the canvas within the defined size
        centre_x = np.random.uniform(-canvas_size / 1.5, canvas_size / 1.5)
        centre_y = np.random.uniform(-canvas_size / 1.5, canvas_size / 1.5)

        # Increase ellipticity and Einstein radius to boost lensing strength
        ell_comps = (np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4))
        einstein_radius = np.random.uniform(1.5, 2.5)  # Higher Einstein radius for even stronger lensing

        # Create a mass profile with the increased Einstein radius
        mass_profile = al.mp.Isothermal(
            centre=(centre_x, centre_y),
            ell_comps=ell_comps,
            einstein_radius=einstein_radius
        )

        # Light profile with random properties for appearance
        intensity = np.random.uniform(0.4, 0.8)
        effective_radius = np.random.uniform(0.3, 0.6)

        # Create a galaxy with the chosen mass and light profiles
        galaxy = al.Galaxy(
            redshift=redshift,
            mass=mass_profile,
            light=al.lp.ExponentialSph(
                centre=(centre_x, centre_y),
                intensity=intensity,
                effective_radius=effective_radius
            )
        )
        galaxies.append(galaxy)
    return galaxies

# Increase the number of galaxies in the cluster and canvas size
n_galaxies = 10
canvas_size = 30.0  # Large canvas size to spread galaxies

# Generate the galaxy cluster with significantly stronger lensing properties
cluster_galaxies = generate_stronger_lensing_galaxy_cluster(
    n=n_galaxies,
    canvas_size=canvas_size
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
    shape_native=(300, 300),
    pixel_scales=0.23  # Pixel scale for capturing fine details
)

# Create a tracer for just the galaxy cluster (lens galaxies only, without the source)
tracer_cluster_only = al.Tracer(galaxies=cluster_galaxies)

# Generate the image of the galaxy cluster alone
cluster_image = tracer_cluster_only.image_2d_from(grid=grid)

# Plot the light from the galaxy cluster alone
cluster_plotter = aplt.Array2DPlotter(array=cluster_image)
cluster_plotter.figure_2d()

# Create a tracer including both the galaxy cluster and the lensed source galaxy
tracer_with_source = al.Tracer(galaxies=cluster_galaxies + [source_galaxy])

# Generate the full lensed image including both the galaxy cluster and the lensed source galaxy
lensed_image_with_cluster = tracer_with_source.image_2d_from(grid=grid)

# Combine the two images by adding the cluster light to the lensed image
combined_image = lensed_image_with_cluster + cluster_image

# Plot the combined image that includes the brighter galaxy cluster and the lensed source galaxy
combined_image_plot = aplt.Array2DPlotter(array=combined_image)
combined_image_plot.figure_2d()

