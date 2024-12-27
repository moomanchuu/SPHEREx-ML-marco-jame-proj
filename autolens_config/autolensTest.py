# # # # # # # # # # # # # # # # 
#
# The general procedure: 
# -> Create lens from a MassProfile
# -> Create lensed galaxy from LightProfile
# -> Define redshifts for both the lens and lensed galaxies
# -> Define how light is deflected otw to Earth by input (ie Planck15)
# -> Now with Tracer and Grid object you can perform calculations :D
#
#
# pixel_scales affects how large the graph is, ie can zoom out/in
# einstein_radius is effective radius of the arc created by lens (I think?)
# 
# EOF
import autolens as al
import autolens.plot as aplt
from astropy import cosmology as cosmo
import random

# To describe the deflection of light by mass, two-dimensional grids of (y,x) Cartesian
# coordinates are used.

grid = al.Grid2D.uniform(
    shape_native=(50, 50),
    pixel_scales=0.1,  # <- Conversion from pixel units to arc-seconds.
)

# The lens galaxy has an elliptical isothermal mass profile and is at redshift 0.5.

def mass():
    # m = al.mp.gNFW(
    #         centre=(random.random(),random.random()), ell_comps=(0.1,0.05), scale_radius=90
    # )
    m = al.mp.Isothermal(
        centre=(random.random(), random.random()), ell_comps=(0.1, 0.05), einstein_radius=1.2
    )
    return m
mass1 = al.mp.Isothermal(
   centre=(0.2, 0.3), ell_comps=(0.1, 0.05), einstein_radius=1.2
)
mass2 = al.mp.Isothermal(
    centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=1.2
)

# m = al.mp.gNFW(
#     centre=(random.random(),random.random()), ell_comps=(0.1,0.05), scale_radius=90
# )

g1 = al.Galaxy(redshift=0.5, mass=mass1)
g2 = al.Galaxy(redshift=0.5, mass=mass2)

lens_tot = al.Galaxies(galaxies = [g1,g2])

# Plot the galaxies
image_galx = lens_tot.image_2d_from(grid=grid)
galx_plot = aplt.GalaxiesPlotter(
        galaxies=lens_tot, grid=grid, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
galx_plot.figures_2d(image=True)


# The source galaxy has an elliptical exponential light profile and is at redshift 1.0
# disk = al.lp.Exponential(
#     centre=(0.3, 0.2),
#     ell_comps=(0.05, 0.25),
#     intensity=0.05,
#     effective_radius=0.5,
# )
# 
# source_galaxy = al.Galaxy(redshift=1.0, disk=disk)


#We create the strong lens using a Tracer, which uses the galaxies, their redshifts
#and an input cosmology to determine how light is deflected on its path to Earth.

# tracer = al.Tracer(
#     galaxies=[lens_tot, source_galaxy],
#     cosmology = al.cosmo.Planck15()
# )
# 
# #We can use the Grid2D and Tracer to perform many lensing calculations, for example
# #plotting the image of the lensed source.
# 
# tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
# tracer_plotter.figures_2d(image=True)

#lensed_image = tracer.image_2d_from(grid=grid)

#mat_plot_2d = aplt.MatPlot2D(
#        title = aplt.Title(label = 'Lensed Image'),
#        axis = aplt.Axis(extent = None),
#        cmap = aplt.Cmap(),
#        figure = aplt.Figure(),
#        output = aplt.Output(),
#        critical_curves = None,
#)

#lensed_image_plotter = aplt.Array2DPlotter(array = lensed_image, mat_plot_2d = mat_plot_2d)
#lensed_image_plotter.figure_2d()
