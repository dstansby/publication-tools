from sunpy.coordinates import frames as sunframes
from astropy.coordinates import SkyCoord
import astropy.units as u


def project_to_ss(coords, vsw, source_surface_r):
    """
    Project a set of abitrary coordinates in the heliosphere on to the source
    surface.

    Parameters
    ----------
    coords : astropy.coordinates.SkyCoord
        Coordinates to be projected onto the source surface.
    vsw : Quantity
        Solar wind velocity used to ballistically project backwards.
    source_surface_r : astropy.units.Quantity
        Source surface radius to project on to.

    Returns
    -------
    seeds_ss : astropy.coordinates.SkyCoord
        Seed points projected on to the source surface.
    """
    coords.representation_type = 'spherical'
    # Calculate the time it takes for the solar wind to travel radially to
    # the source surface
    dt = ((coords.radius - source_surface_r) / vsw).to(u.s)
    # Construct the Carrington frame that existed when the plasma left the
    # source surface
    ss_frame = sunframes.HeliographicCarrington(obstime=coords.obstime - dt)
    # Transform to this frame
    coords_ss = coords.transform_to(ss_frame)
    # Finally, set the radius to the source surface
    coords_ss = SkyCoord(
        coords_ss.lon,
        coords_ss.lat,
        source_surface_r,
        obstime=coords.obstime,
        observer=coords.observer,
        frame='heliographic_carrington')
    return coords_ss
