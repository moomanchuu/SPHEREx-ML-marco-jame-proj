import autolens as al
import autolens.plot as aplt
import numpy as np

def generate_gaussian_distributed_galaxy_cluster(n, center=(0.0, 0.0), std_dev=8.5, redshift=0.5):
    """Generates a list of n galaxies distributed around a center with Gaussian spread."""
    galaxies = []
    for _ in range(n):
        # Randomly assign galaxy position based on a Gaussian distribution
        centre_x = np.random.normal(loc=center[0], scale=std_dev)  # x position
        centre_y = np.random.normal(loc=center[1], scale=std_dev)  # y position

        # Other random galaxy parameters
        ell_comps = (np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2))  # Ellipticity
        einstein_radius = np.random.uniform(1.5, 3.5)  # Increased Einstein radius
        intensity = np.random.uniform(0.3, 0.6)  # Intensity
        effective_radius = np.random.uniform(0.3, 0.5)  # Effective radius

        # Create a galaxy with mass and light profiles
        galaxy = al.Galaxy(
            redshift=redshift,
            mass=al.mp.Isothermal(
                centre=(centre_x, centre_y),
                ell_comps=ell_comps,
                einstein_radius=einstein_radius
            ),
            light=al.lp.ExponentialSph(
                centre=(centre_x, centre_y),
                intensity=intensity,
                effective_radius=effective_radius
            )
        )
        galaxies.append(galaxy)
    return galaxies

def random_position_within_circle(radius_min=1.0, radius_max=8.5):
    """Generate a random position within a circle centered at the origin with a minimum and maximum radius."""
    # Random radius between the specified minimum and maximum radii
    r = np.random.uniform(radius_min, radius_max)
    # Random angle between 0 and 2*pi
    theta = np.random.uniform(0, 2 * np.pi)
    # Convert polar to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x, y)

# Define the number of galaxies in the cluster and Gaussian distribution parameters
n_galaxies = 10
cluster_center = (0.0, 0.0)  # Center of the galaxy cluster
position_std_dev = 8.5  # Increased standard deviation for wider spread

# Generate the galaxy cluster with n galaxies distributed according to a Gaussian
cluster_galaxies = generate_gaussian_distributed_galaxy_cluster(
    n=n_galaxies,
    center=cluster_center,
    std_dev=position_std_dev
)

# Generate a random position for the source galaxy within a circle of radius 8.5, with a minimum radius to avoid overlap
source_position = random_position_within_circle(radius_min=1.0, radius_max=8.5)

# Define the source galaxy (the galaxy being lensed) with the random position
source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.ExponentialSph(
        centre=source_position,
        intensity=0.5,
        effective_radius=0.3
    )
)

# Define a larger grid for the simulation with increased arcsecond scale and a larger field of view
grid = al.Grid2D.uniform(
    shape_native=(300, 300),  # Increased grid size
    pixel_scales=0.15  # Adjusted pixel scale to maintain a detailed view
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

