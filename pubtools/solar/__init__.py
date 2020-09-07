import astropy.constants as const
import astropy.units as u
import numpy as np
import sunpy.coordinates
import sunpy.map


def contour_to_mask(ctr, m, upsample=1):
    """
    Given set of coordinates definining a contour in world coordinates,
    create a mask that is 1 inside the contour and 0 outside.

    Parameters
    ----------
    ctr : astropy.coordinates.SkyCoord
    m : sunpy.map.Map

    Returns
    -------
    mask : sunpy.map.Map
    """
    from skimage import draw
    # Get pixel values of the contour
    ch_indices = np.array(m.wcs.world_to_pixel(ctr)).T
    mask = draw.polygon2mask(m.data.shape, ch_indices)
    return mask


def aia_prep(aia_map):
    """
    Prep an AIA map.

    This runs:
    - `aiapy.calibrate.update_pointing`
    - `aiapy.calibrate.register`
    - `calibrate.correct_degradation(aia_map)`

    And normalises the data by exposure time to get the data in units of DN/s.

    Parameters
    ----------
    aia_map : sunpy.map.sources.AIAMap
        A full disc AIA map.

    Returns
    -------
    aia_map : sunpy.map.sources.AIAMap
        Prepped map.
    """
    from aiapy import calibrate
    aia_map = calibrate.update_pointing(aia_map)
    aia_map = calibrate.register(aia_map)
    aia_map = calibrate.correct_degradation(aia_map)
    # Convert from DN to DN/s
    aia_map = sunpy.map.Map(
        aia_map.data / aia_map.exposure_time.to_value(u.s),
        aia_map.meta
    )
    return aia_map


def cos_theta(m):
    r"""
    Return :math:`\cos \theta` at all pixel centers, where :math:`\theta` is
    the angle on the photsphere away from the sub-osberver line measured
    about the solar center.

    Parameters
    ----------
    m : sunpy.map.Map
    """
    # The z-axis is aligned with the sun-observer line
    z = sunpy.map.all_coordinates_from_map(m).transform_to('heliocentric').z
    return (z / const.R_sun).value


def cos_theta_correction(m):
    r"""
    Divide the data in *m* by :math:`\cos \theta`.

    This is commonly used with LOS magnetograms to get the radial field,
    assuming the transverse components are zero.

    Returns
    -------
    sunpy.map.GenericMap
        Map normalised by 1/cos(theta)
    """
    with np.errstate(divide='ignore'):
        data = m.data / cos_theta(m)
    m_corrected = sunpy.map.Map(data, m.meta)
    return m_corrected


def deprojected_areas(m):
    """
    Return the deprojected areas of each pixel in *m*.

    Uses the small angle approximation, ie. each pixel width and height is
    much smaller than 1 (in radians).
    """
    return (m.scale.axis1 * 1 * u.pix *
            m.scale.axis2 * 1 * u.pix *
            m.observer_coordinate.radius**2 / cos_theta(m) / u.rad**2).to(u.m**2)


def contour(m, level):
    """
    Returns coordinates of the contours for a given level value.

    Parameters
    ----------
    m : `~sunpy.map.GenericMap`
        Input map.
    level : float
        Value along which to find contours in the array.

    Returns
    -------
    contours: list of (n,2) `~astropy.units.Quantity`

        Each contour is an ndarray of shape (n, 2), consisting of n
        (row, column) coordinates along the contour.
    """
    from skimage import measure
    contours = measure.find_contours(m.data, level=level)
    contours = [m.wcs.array_index_to_world(c[:, 0], c[:, 1]) for c in contours]
    return contours


def remove_obs_keywords(meta):
    """
    Parameters
    ----------
    meta : dict
    """
    for key in ['crln_obs', 'crlt_obs', 'dsun_obs',
                'hgln_obs', 'hglt_obs', 'dsun_obs']:
        meta.pop(key, None)
    return meta


def set_observer_coord(m, observer_coord):
    """
    Set observer coordinate.

    Parameters
    ----------
    m : `~sunpy.map.GenericMap`
        Input map.
    observer_coord : astropy.coordiantes.SkyCoord
        Observer coordinate.
    """
    from sunpy.coordinates import frames
    new_obs_frame = frames.HeliographicStonyhurst(obstime=m.date)
    observer_coord = observer_coord.transform_to(new_obs_frame)

    new_meta = remove_obs_keywords(m.meta)
    new_meta['hglt_obs'] = observer_coord.lat.to_value(u.deg)
    new_meta['hgln_obs'] = observer_coord.lon.to_value(u.deg)
    new_meta['dsun_obs'] = observer_coord.radius.to_value(u.m)
    return sunpy.map.Map(m.data, new_meta)


def set_earth_obs_coord(m):
    """
    Set the observer coordinate of *m* to Earth.

    Parameters
    ----------
    m : `~sunpy.map.GenericMap`
        Input map.
    """
    return set_observer_coord(m, sunpy.coordinates.get_earth(m.date))
