import autolens as al
import autolens.plot as aplt

# Define the lens galaxy cluster
lens_galaxy_1 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, -1.0),
        ell_comps=(0.1, 0.05),
        einstein_radius=1.2
    )
)

lens_galaxy_2 = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(1.0, 1.0),
        ell_comps=(0.05, -0.05),
        einstein_radius=0.8
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

# Generate the convergence map for the galaxy cluster (lens)
convergence = tracer.convergence_2d_from(grid=grid)

# Plot the convergence map of the galaxy cluster
convergence_plotter = aplt.Array2DPlotter(array=convergence)
convergence_plotter.figure_2d()

# Generate the lensed image of the source galaxy
lensed_image = tracer.image_2d_from(grid=grid)

# Plot the lensed image
lensed_image_plot = aplt.Array2DPlotter(array=lensed_image)
lensed_image_plot.figure_2d()

