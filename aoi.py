"""Shared area-of-interest and target-grid definitions.

Two ~45 km boxes in Laikipia, Kenya. Box A is development; Box B is the held-out
landscape and is never used for fitting. All rasters are warped onto a fixed
EPSG:4326 grid at RES degrees so every source shares pixel indices.
"""

# (west, south, east, north) in degrees
BOXES = {
    "A": (37.15, 0.05, 37.55, 0.45),  # Lewa / Borana / Il Ngwesi / Mukogodo
    "B": (36.70, 0.25, 37.10, 0.65),  # Mpala / Ol Jogi / Loisaba / Koija (held out)
}

RES = 0.0025  # degrees, ~278 m at the equator; close to MODIS 250 m
N = 160  # pixels per side: 0.40 / 0.0025

# Split policy (years are calendar years; MODIS composites are dated by start day)
TRAIN_YEARS = range(2003, 2018)
VAL_YEARS = [2018]
TEST_YEARS = range(2019, 2026)
CLIM_YEARS = range(2003, 2019)  # climatology and anomaly baseline: train + val only
DEV_BOX = "A"
HELD_OUT_BOX = "B"


def grid_transform(box):
    """Affine transform of the target grid for a box (north-up, origin top-left)."""
    from rasterio.transform import from_origin

    w, s, e, n = BOXES[box]
    return from_origin(w, n, RES, RES)


def assert_allowed_for_training(box, years):
    """Refuse held-out data in any fitting call."""
    if box == HELD_OUT_BOX:
        raise ValueError("Box B is the held-out landscape; it must not be used for fitting.")
    bad = [y for y in years if y not in TRAIN_YEARS and y not in VAL_YEARS]
    if bad:
        raise ValueError(f"Years {bad} are held out; they must not be used for fitting.")
