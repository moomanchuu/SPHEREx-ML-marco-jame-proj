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
    """
    Generate a collection of galaxy cluster objects, including:
      - A large EllNFW dark matter halo at (possibly offset) cluster center.
      - External shear for the cluster environment.
      - Subhalos scattered around (optional).
      - Several cluster member galaxies with elliptical mass (EllIsothermal) 
        and elliptical Sersic light (EllSersic).
    """

    galaxies = []

    # -----------------------------------------------------------------
    # 1) CLUSTER HALO: EllNFW (using ell_comps)
    # -----------------------------------------------------------------
    # Randomly shift the cluster center from (0, 0)
    cluster_x = np.random.uniform(-cluster_center_offset, cluster_center_offset)
    cluster_y = np.random.uniform(-cluster_center_offset, cluster_center_offset)

    cluster_halo = al.mp.gNFW(
        centre=(cluster_x, cluster_y),
        ell_comps=(
            np.random.uniform(-0.2, 0.2),  # controls ellipticity & angle
            np.random.uniform(-0.2, 0.2),
        ),
        kappa_s=np.random.uniform(0.1, 0.3),       # dimensionless amplitude
        scale_radius=np.random.uniform(20.0, 40.0) # scale radius in arcsec
    )

    # -----------------------------------------------------------------
    # 2) EXTERNAL SHEAR
    # -----------------------------------------------------------------
    external_shear = al.mp.ExternalShear(
        gamma_1=np.random.uniform(-0.05, 0.05),
        gamma_2=np.random.uniform(-0.05, 0.05)
    )

    # The cluster "galaxy": combine halo + external shear under one Galaxy object
    cluster_galaxy = al.Galaxy(
        redshift=cluster_central_redshift,
        mass=cluster_halo,
        shear=external_shear
    )
    galaxies.append(cluster_galaxy)

    # -----------------------------------------------------------------
    # 3) OPTIONAL SUBHALOS
    # -----------------------------------------------------------------
    if with_subhalos:
        for _ in range(n_subhalos):
            # Place subhalo within +/- galaxy_spread from cluster center
            subhalo_x = np.random.uniform(-galaxy_spread, galaxy_spread)
            subhalo_y = np.random.uniform(-galaxy_spread, galaxy_spread)

            subhalo_mass = al.mp.gNFW(
                centre=(subhalo_x, subhalo_y),
                kappa_s=np.random.uniform(0.01, 0.05),
                scale_radius=np.random.uniform(1.0, 5.0)
            )

            subhalo_galaxy = al.Galaxy(
                redshift=cluster_central_redshift,
                mass=subhalo_mass
            )
            galaxies.append(subhalo_galaxy)

    # -----------------------------------------------------------------
    # 4) CLUSTER MEMBER GALAXIES
    # -----------------------------------------------------------------
    for _ in range(n_members):
        # Slight randomization in redshift around the cluster center
        gal_redshift = np.random.normal(loc=cluster_central_redshift, scale=0.01)

        # Random XY in the cluster region
        gx = np.random.uniform(-galaxy_spread, galaxy_spread)
        gy = np.random.uniform(-galaxy_spread, galaxy_spread)

        # Elliptical isothermal mass profile
        mass_profile = al.mp.Isothermal(
            centre=(gx, gy),
            ell_comps=(
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3)
            ),
            einstein_radius=np.random.uniform(1.0, 3.0)
        )

        # Elliptical Sersic light profile
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
    """
    Generates a background source galaxy with an elliptical Sersic light profile.
    """

    # Elliptical Sersic light for the source
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
    plot_result=True
):
    """
    Creates a realistic lensing simulation of a galaxy cluster, including:
      - EllNFW cluster halo + external shear
      - Subhalos (SphNFW)
      - Member galaxies (EllIsothermal + EllSersic)
      - A background source galaxy (EllSersic)
      - Convolution with a PSF and addition of Poisson noise
    """

    # -----------------------------------------------------------------
    # 1) Generate cluster galaxies (including halo, subhalos, etc.)
    # -----------------------------------------------------------------
    cluster_galaxies = generate_realistic_cluster_galaxies(
        n_members=n_cluster_members,
        cluster_central_redshift=cluster_central_redshift,
        with_subhalos=True,
        n_subhalos=2
    )

    # -----------------------------------------------------------------
    # 2) Generate the background source galaxy
    # -----------------------------------------------------------------
    source_galaxy = generate_source_galaxy(
        source_redshift=source_redshift
    )

    # -----------------------------------------------------------------
    # 3) Create a grid covering our field of view
    # -----------------------------------------------------------------
    grid = al.Grid2D.uniform(
        shape_native=(200, 200),
        pixel_scales=0.1
    )

    # -----------------------------------------------------------------
    # 4) Make a Tracer from the cluster + source
    # -----------------------------------------------------------------
    tracer = al.Tracer(galaxies=cluster_galaxies + [source_galaxy])

    # -----------------------------------------------------------------
    # 5) Create a simple Gaussian PSF & set up the SimulatorImaging
    # -----------------------------------------------------------------
    psf = al.Kernel2D.from_gaussian(
        shape_native=(21, 21),
        sigma=0.8,  # in pixels
        pixel_scales=grid.pixel_scales
    )

    simulator = al.SimulatorImaging(
        exposure_time=300.0,       # seconds
        background_sky_level=1.0,  # average sky background level
        psf=psf,
        add_poisson_noise=True,
        noise_seed=1
    )

    # -----------------------------------------------------------------
    # 6) Generate the simulated imaging (image + noise map + etc.)
    # -----------------------------------------------------------------
    imaging = simulator.via_tracer_from(
        tracer=tracer,
        grid=grid
    )

    # -----------------------------------------------------------------
    # 7) Plot results (optional)
    # -----------------------------------------------------------------
    if plot_result:
        imaging_plotter = aplt.ImagingPlotter(dataset=imaging)
        imaging_plotter.subplot_dataset()

        tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
        tracer_plotter.subplot_tracer()

    return imaging


def wrapperFunction(
    n_cluster_members=5,
    cluster_central_redshift=0.5,
    source_redshift=1.5,
    plot_result=True
):
    """
    Wrapper function to be called externally (e.g., from a Jupyter notebook).
    Returns the simulated PyAutoLens Imaging object.
    """
    simulated_data = simulate_realistic_cluster_lensing(
        n_cluster_members=n_cluster_members,
        cluster_central_redshift=cluster_central_redshift,
        source_redshift=source_redshift,
        plot_result=plot_result
    )
    return simulated_data


if __name__ == "__main__":
    # Example usage / test run
    data = wrapperFunction(
        n_cluster_members=5,
        cluster_central_redshift=0.5,
        source_redshift=1.5,
        plot_result=True
    )

    # 'data' is a PyAutoLens Imaging object with .image, .noise_map, .psf, etc.
    # You can proceed with further modeling or analysis as needed.
