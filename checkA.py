import argparse
import rasterio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geotiff", required=True)
    args = ap.parse_args()

    points = [
        ("Monserrate", -74.0555, 4.6057),
        ("Cable",      -74.0507, 4.6297),
        ("Aguanoso",   -74.0531, 4.5756),
    ]

    with rasterio.open(args.geotiff) as ds:
        T = ds.transform
        lon0, lat0 = float(T.c), float(T.f)
        dlon, dlat = float(T.a), float(T.e)

    def lonlat_to_xy(lon, lat):
        x = (lon - lon0) / dlon
        y = (lat - lat0) / dlat
        return x, y

    def xy_to_lonlat(x, y):
        lon = lon0 + x * dlon
        lat = lat0 + y * dlat
        return lon, lat

    print("[GeoTIFF]")
    print("  lon0, lat0 =", lon0, lat0)
    print("  dlon, dlat =", dlon, dlat)
    print()

    for name, lon, lat in points:
        x, y = lonlat_to_xy(lon, lat)
        lon2, lat2 = xy_to_lonlat(x, y)
        print(name)
        print("  lon,lat        =", lon, lat)
        print("  x,y (float)     =", x, y)
        print("  back lon,lat    =", lon2, lat2)
        print("  error (lon,lat) =", lon2 - lon, lat2 - lat)
        print()

if __name__ == "__main__":
    main()
