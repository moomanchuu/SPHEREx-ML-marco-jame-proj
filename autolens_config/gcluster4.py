import autolens as al
import autolens.plot as aplt
import numpy as np

# Define the lens galaxy cluster with both mass and light profiles
lens_galaxy_1 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, -1.0),
        ell_comps=(0.1, 0.05),
        einstein_radius=1.2
    ),
    light=al.lp.ExponentialSph(
        centre=(0.0, -1.0),
        intensity=0.3,
        effective_radius=0.2
    )
)

lens_galaxy_2 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(1.0, 1.0),
        ell_comps=(0.05, -0.05),
        einstein_radius=0.8
    ),
    light=al.lp.ExponentialSph(
        centre=(1.0, 1.0),
        intensity=0.3,
        effective_radius=0.2
    )
)

# Define the source galaxy (the galaxy being lensed)
source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.ExponentialSph(
        centre=(0.0, 0.0),
        intensity=0.5,
        effective_radius=0.1
    )
)

# Define a grid for the simulation
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05
)

# Create a tracer for just the galaxy cluster (lens galaxies only, without the source)
tracer_cluster_only = al.Tracer(galaxies=[lens_galaxy_1, lens_galaxy_2])

# Generate the image of the galaxy cluster alone
cluster_image = tracer_cluster_only.image_2d_from(grid=grid)

# Plot the light from the galaxy cluster alone
cluster_plotter = aplt.Array2DPlotter(array=cluster_image)
cluster_plotter.figure_2d()

# Create a tracer including both the galaxy cluster and the lensed source galaxy
tracer_with_source = al.Tracer(galaxies=[lens_galaxy_1, lens_galaxy_2, source_galaxy])

# Generate the full lensed image including both the galaxy cluster and the lensed source galaxy
lensed_image_with_cluster = tracer_with_source.image_2d_from(grid=grid)

# Combine the two images by adding the cluster light to the lensed image
combined_image = lensed_image_with_cluster + cluster_image

# Plot the combined image that includes the brighter galaxy cluster and the lensed source galaxy
combined_image_plot = aplt.Array2DPlotter(array=combined_image)
combined_image_plot.figure_2d()

