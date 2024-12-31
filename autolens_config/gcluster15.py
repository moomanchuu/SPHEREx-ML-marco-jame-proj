import numpy as np
import autolens as al
import autolens.plot as aplt

def generate_realistic_cluster_galaxies(
    n_members=5,
    cluster_center_offset=5.0,
    cluster_central_redshift=0.5,
    galaxy_spread=10.0,
    with_subhalos=True,
    n_subhalos=2
):
    galaxies = []

    # 1) CLUSTER HALO: gNFW
    cluster_x = np.random.uniform(-cluster_center_offset, cluster_center_offset)
    cluster_y = np.random.uniform(-cluster_center_offset, cluster_center_offset)

    cluster_halo = al.mp.gNFW(
        centre=(cluster_x, cluster_y),
        ell_comps=(
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.2, 0.2),
        ),
        kappa_s=np.random.uniform(0.1, 0.3),      
        scale_radius=np.random.uniform(30.0, 45.0),
        inner_slope=1.0
    )

    # 2) EXTERNAL SHEAR
    external_shear = al.mp.ExternalShear(
        gamma_1=np.random.uniform(-0.05, 0.05),
        gamma_2=np.random.uniform(-0.05, 0.05)
    )

    # Combined cluster "galaxy"
    cluster_galaxy = al.Galaxy(
        redshift=cluster_central_redshift,
        mass=cluster_halo,
        shear=external_shear
    )
    galaxies.append(cluster_galaxy)

    # 3) SUBHALOS
    if with_subhalos:
        for _ in range(n_subhalos):
            subhalo_x = np.random.uniform(-galaxy_spread, galaxy_spread)
            subhalo_y = np.random.uniform(-galaxy_spread, galaxy_spread)

            subhalo_mass = al.mp.gNFW(
                centre=(subhalo_x, subhalo_y),
                ell_comps=(
                    np.random.uniform(-0.2, 0.2),
                    np.random.uniform(-0.2, 0.2)
                ),
                kappa_s=np.random.uniform(0.01, 0.05),
                scale_radius=np.random.uniform(1.0, 5.0),
                inner_slope=1.0
            )

            subhalo_galaxy = al.Galaxy(
                redshift=cluster_central_redshift,
                mass=subhalo_mass
            )
            galaxies.append(subhalo_galaxy)

    # 4) CLUSTER MEMBER GALAXIES
    for _ in range(n_members):
        gal_redshift = np.random.normal(loc=cluster_central_redshift, scale=0.01)

        gx = np.random.uniform(-galaxy_spread, galaxy_spread)
        gy = np.random.uniform(-galaxy_spread, galaxy_spread)

        mass_profile = al.mp.Isothermal(
            centre=(gx, gy),
            ell_comps=(
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3)
            ),
            einstein_radius=np.random.uniform(1.0, 3.0)
        )

        bulge_profile = al.lp.Sersic(
            centre=(gx, gy),
            ell_comps=(
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3)
            ),
            intensity=np.random.uniform(0.3, 1.0),
            effective_radius=np.random.uniform(0.5, 1.5),
            sersic_index=np.random.uniform(1.0, 4.0)
        )

        galaxy_member = al.Galaxy(
            redshift=gal_redshift,
            mass=mass_profile,
            bulge=bulge_profile
        )
        galaxies.append(galaxy_member)

    return galaxies


def generate_source_galaxy(
    source_redshift=1.5,
    source_centre=(0.0, 0.0)
):
    bulge_profile = al.lp.Sersic(
        centre=source_centre,
        ell_comps=(
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.3, 0.3)
        ),
        intensity=np.random.uniform(0.3, 1.0),
        effective_radius=np.random.uniform(0.2, 1.0),
        sersic_index=np.random.uniform(0.8, 2.0)
    )

    source_galaxy = al.Galaxy(
        redshift=source_redshift,
        bulge=bulge_profile
    )

    return source_galaxy


def simulate_realistic_cluster_lensing(
    n_cluster_members=5,
    cluster_central_redshift=0.5,
    source_redshift=1.5,
    plot_result=True,
    shape_native=(683,683), 
    pixel_scales=0.2
):
    cluster_galaxies = generate_realistic_cluster_galaxies(
        n_members=n_cluster_members,
        cluster_central_redshift=cluster_central_redshift,
        with_subhalos=True,
        n_subhalos=2
    )

    source_galaxy = generate_source_galaxy(
        source_redshift=source_redshift
    )

    grid = al.Grid2D.uniform(
        shape_native=shape_native,
        pixel_scales=pixel_scales
    )
    tracer = al.Tracer(galaxies=cluster_galaxies + [source_galaxy])

    psf = al.Kernel2D.from_gaussian( #Assuming this PSF is << SPHEREx PSF....
        shape_native=(21, 21),
        sigma=0.8,
        pixel_scales=grid.pixel_scales
    )

    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        background_sky_level=1.0,
        psf=psf,
        add_poisson_noise=True,
        noise_seed=1
    )

    imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

    if plot_result:
        imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
        imaging_plotter.figures_2d(data=True)
        # to show all subplots just do .subplot_dataset(), or see gcluster13

    return imaging

def simulate_realistic_cluster_no_lensing( #Marco lazy and doesnt want to mess with the already working code
    n_cluster_members=5,
    cluster_central_redshift=0.5,
    plot_result=True,
    shape_native=(683, 683), 
    pixel_scales=0.2
):
    cluster_galaxies = generate_realistic_cluster_galaxies(
        n_members=n_cluster_members,
        cluster_central_redshift=cluster_central_redshift,
        with_subhalos=True,
        n_subhalos=2
    )

    grid = al.Grid2D.uniform(
        shape_native=shape_native,
        pixel_scales=pixel_scales
    )

    psf = al.Kernel2D.from_gaussian(  # Assuming this PSF is << SPHEREx PSF
        shape_native=(21, 21),
        sigma=0.8,
        pixel_scales=grid.pixel_scales
    )

    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        background_sky_level=1.0,
        psf=psf,
        add_poisson_noise=True,
        noise_seed=1
    )

    imaging = simulator.via_galaxies_from(galaxies=cluster_galaxies, grid=grid)

    if plot_result:
        imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
        imaging_plotter.figures_2d(data=True)

    return imaging


def wrapperFunction(
    n_cluster_members=5,
    cluster_central_redshift=0.5,
    source_redshift=1.5,
    lensing = 1,
    plot_result=True,
):
    
    if lensing == 1:
        return simulate_realistic_cluster_lensing(
            n_cluster_members=n_cluster_members,
            cluster_central_redshift=cluster_central_redshift,
            source_redshift=source_redshift,
            plot_result=plot_result
        )
    if lensing == 0: 
        return simulate_realistic_cluster_no_lensing(
            n_cluster_members=n_cluster_members,
            cluster_central_redshift=cluster_central_redshift,
            plot_result=plot_result
        )

if __name__ == "__main__":
    data = wrapperFunction(
        n_cluster_members=5,
        cluster_central_redshift=0.5,
        source_redshift=1.5,
        lensing = 1,
        plot_result=True

    )
