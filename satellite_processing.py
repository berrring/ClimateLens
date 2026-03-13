import numpy as np

EPSILON = 1e-6


def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + EPSILON)


def calculate_ndwi(green, nir):
    return (green - nir) / (green + nir + EPSILON)


def calculate_ndbi(swir, nir):
    return (swir - nir) / (swir + nir + EPSILON)


def calculate_indices(red, green, nir, swir):
    ndvi = calculate_ndvi(nir, red)
    ndwi = calculate_ndwi(green, nir)
    ndbi = calculate_ndbi(swir, nir)
    return {
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndbi": ndbi,
    }


def calculate_brightness(red, green, blue):
    return (red + green + blue) / 3.0


def classify_landcover(red, green, blue, nir, swir):
    indices = calculate_indices(red, green, nir, swir)
    brightness = calculate_brightness(red, green, blue)

    ndvi = indices["ndvi"]
    ndwi = indices["ndwi"]
    ndbi = indices["ndbi"]

    vegetation = ndvi > 0.3
    water = ndwi > 0.2
    ice = (brightness > 0.7) & (ndvi < 0.2)
    urban = (ndbi > 0.1) & (ndvi < 0.2) & (~water) & (~ice)

    classes = {
        "vegetation": vegetation,
        "water": water,
        "urban": urban,
        "ice": ice,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndbi": ndbi,
        "brightness": brightness,
    }
    return classes


def percent_mask(mask):
    return float(mask.sum()) / float(mask.size) * 100.0


def stats_from_classes(classes):
    return {
        "vegetation": percent_mask(classes["vegetation"]),
        "water": percent_mask(classes["water"]),
        "urban": percent_mask(classes["urban"]),
        "ice": percent_mask(classes["ice"]),
        "ndvi_mean": float(np.mean(classes["ndvi"])),
        "ndwi_mean": float(np.mean(classes["ndwi"])),
        "ndbi_mean": float(np.mean(classes["ndbi"])),
    }


def change_maps(classes_start, classes_end):
    return {
        key: classes_end[key].astype(np.int8) - classes_start[key].astype(np.int8)
        for key in ["vegetation", "water", "urban", "ice"]
    }
