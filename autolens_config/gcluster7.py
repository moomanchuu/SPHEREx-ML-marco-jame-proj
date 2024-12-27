import autolens as al
import autolens.plot as aplt
import numpy as np

def generate_irregular_galaxy_cluster(n, canvas_size=30.0, redshift=0.5):
    """Generates a list of n galaxies with varied ellipticity and mass profiles randomly distributed across the canvas."""
    galaxies = []
    for _ in range(n):
        # Random positions across the canvas within the defined size
        centre_x = np.random.uniform(-canvas_size / 2, canvas_size / 2)
        centre_y = np.random.uniform(-canvas_size / 2, canvas_size / 2)

        # Randomize ellipticity and limit Einstein radii for varied lensing strengths
        ell_comps = (np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3))
        einstein_radius = np.random.uniform(1.5, 2.5)  # Limited Einstein radius for weaker lensing

        # Create a mass profile with varied ellipticity and Einstein radius
        mass_profile = al.mp.Isothermal(
            centre=(centre_x, centre_y),
            ell_comps=ell_comps,
            einstein_radius=einstein_radius
        )

        # Light profile with random properties for appearance
        intensity = np.random.uniform(0.2, 0.6)
        effective_radius = np.random.uniform(0.2, 0.5)

        # Create a galaxy with the selected mass and light profiles
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

# Number of galaxies in the cluster and canvas size for random distribution
n_galaxies = 8
canvas_size = 30.0  # Canvas size to cover a wide area

# Generate the galaxy cluster with irregular mass profiles and random positions
cluster_galaxies = generate_irregular_galaxy_cluster(
    n=n_galaxies,
    canvas_size=canvas_size
)

# Position the source galaxy closer to the cluster center for effective lensing, with slight offset
source_position = np.random.uniform(-5.0, 5.0, size=2)  # Random position within 5 arcseconds of center

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.ExponentialSph(
        centre=(np.random.uniform(-5.0,5.0), np.random.uniform(-5.0,5.0)),
        intensity=0.5,
        effective_radius=0.3
    )
)

# Define a grid with appropriate field of view and detail
grid = al.Grid2D.uniform(
    shape_native=(300, 300),
    pixel_scales=0.2  # Pixel scale to capture fine details
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

