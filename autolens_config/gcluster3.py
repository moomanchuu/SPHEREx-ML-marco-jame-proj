import autolens as al
import autolens.plot as aplt

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

# Create the tracer with the galaxies
tracer = al.Tracer(galaxies=[lens_galaxy_1, lens_galaxy_2, source_galaxy])

# Define a grid for the simulation
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05
)

# Create images of just the lens galaxies (galaxy cluster)
lens_light_image_1 = lens_galaxy_1.image_2d_from(grid=grid)
lens_light_image_2 = lens_galaxy_2.image_2d_from(grid=grid)

# Sum the lens light images to get the complete galaxy cluster image
lens_light_image = lens_light_image_1 + lens_light_image_2

# Plot the light from the galaxy cluster
lens_light_plotter = aplt.Array2DPlotter(array=lens_light_image)
lens_light_plotter.figure_2d()

# Generate the full lensed image including both the galaxy cluster and the lensed source galaxy
lensed_image_with_cluster = tracer.image_2d_from(grid=grid)

# Plot the final image that includes the lensed galaxy and the galaxy cluster
lensed_image_plot = aplt.Array2DPlotter(array=lensed_image_with_cluster)
lensed_image_plot.figure_2d()

