import autolens as al
import autolens.plot as aplt
from astropy import cosmology as cosmo
import random

# To describe the deflection of light by mass, two-dimensional grids of (y,x) Cartesian
# coordinates are used.

grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,  # <- Conversion from pixel units to arc-seconds.
)

sis_mass1 = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6)
sis_mass2 = al.mp.IsothermalSph(centre=(0.2, 0.3), einstein_radius=1.6)

galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
    mass = sis_mass1,
)

extra_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.2, 0.3),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
    mass = sis_mass2
)

galaxies = al.Galaxies(galaxies=[galaxy,extra_galaxy])
image = galaxies.image_2d_from(grid=grid)
galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

# The source galaxy has an elliptical exponential light profile and is at redshift 1.0
disk = al.lp.Exponential(
    centre=(0.3, 0.2),
    ell_comps=(0.05, 0.25),
    intensity=0.05,
    effective_radius=0.5,
)

source_galaxy = al.Galaxy(redshift=1.0, disk=disk)


#We create the strong lens using a Tracer, which uses the galaxies, their redshifts
#and an input cosmology to determine how light is deflected on its path to Earth.

tracer = al.Tracer(
    galaxies=[[galaxy, extra_galaxy], source_galaxy],
    cosmology = al.cosmo.Planck15()
)

#We can use the Grid2D and Tracer to perform many lensing calculations, for example
#plotting the image of the lensed source.
# lensed_image = tracer.image_2d_from(grid=grid)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

