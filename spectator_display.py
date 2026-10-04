"""Display-only sizing; never changes native map coordinates or simulation time."""


def fitted_window_size(desktops, display=0, scale=None, canvas=(1800, 900)):
    if not desktops or not 0 <= display < len(desktops):
        raise ValueError('Select an available display index (0 is the primary display)')
    if scale is not None and not .25 <= scale <= 2.:
        raise ValueError('Window scale must be between 0.25 and 2.0')
    width, height = desktops[display]
    factor = min(1., width*.9/canvas[0], height*.85/canvas[1]) if scale is None else scale
    return tuple(max(1, round(side*factor)) for side in canvas)
