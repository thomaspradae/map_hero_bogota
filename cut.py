import geopandas as gpd
from shapely.ops import unary_union
import unicodedata

IN_PATH = "bogota_localidades.geojson"
OUT_PATH = "aoi_localidades.geojson"

def norm(s: str) -> str:
    s = (s or "").strip()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = s.lower().strip()
    s = s.replace("localidad ", "")  # handles "Localidad Chapinero"
    s = " ".join(s.split())          # collapse repeated whitespace
    return s

# Write KEEP in "human" form, then normalize it.
KEEP_RAW = {
    "chapinero",
    "santa fe",
    "barrios unidos",
    "puente aranda",
    "antonio nariño",        # will become "antonio narino"
    "rafael uribe uribe",    # add
    "teusaquillo",           # add
    "san cristobal",
    "la candelaria",
    "los martires",
}

KEEP = {norm(x) for x in KEEP_RAW}

gdf = gpd.read_file(IN_PATH)
if "name" not in gdf.columns:
    raise SystemExit(f"{IN_PATH} has no 'name' property")

gdf["name_norm"] = gdf["name"].astype(str).map(norm)

sel = gdf[gdf["name_norm"].isin(KEEP)].copy()
if sel.empty:
    raise SystemExit(
        f"Selection empty.\n"
        f"Names present include: {sorted(set(gdf['name_norm']))[:40]} ..."
    )

# Optional debug: show what you failed to match (super useful if spelling differs)
present = set(gdf["name_norm"])
missing_in_file = sorted(KEEP - present)
if missing_in_file:
    print("WARNING: These KEEP names were not found in the file:", missing_in_file)

# Union into one (Multi)Polygon AOI
aoi_geom = unary_union(sel.geometry)

out = gpd.GeoDataFrame(
    [{"name": "AOI_localidades", "geometry": aoi_geom}],
    crs=sel.crs,
)

out.to_file(OUT_PATH, driver="GeoJSON")
print(f"Wrote {OUT_PATH} with 1 feature. Geometry type: {out.geometry.iloc[0].geom_type}")
