class Constants:
    """
    Physical and cosmological constants used by the simulation.

    gee is in kpc*(km/s)**2/Msun. Characteristic simulation conversions use
    kpc for length, Msun for mass, km/s for velocity, and Gyr for time.
    The additional Mpc and cosmological constants support profile scaling.
    Values are class attributes and may be overridden before constructing a State.
    """

    # Gravitational constant in (M_sun^-1 Mpc (km/s)^2)
    gee = 4.2994e-6

    # Conversion of megaparsec to centimeters
    Mpc_to_cm = 3.086e24
    kpc_to_km = 3.086e16

    # Conversion of solar mass to grams
    Msun_to_gram = 1.99e33

    # Conversion of seconds to gigayears
    sec_to_Gyr = 3.16881e-17

    # Cosmological parameters
    xhubble = 0.7           # Hubble constant in units of 100 km/s/Mpc
    Omega_m = 0.3           # Matter density parameter
    Delta_vir = 97.0        # Virial overdensity
