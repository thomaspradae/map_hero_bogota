import osmnx as ox
import geopandas as gpd
import unicodedata
import re
from shapely.ops import unary_union

PLACE = "Bogotá, Colombia"

LOCALIDADES = [
    "Usaquén",
    "Chapinero",
    "Santa Fe",
    "San Cristóbal",
    "Usme",
    "Tunjuelito",
    "Bosa",
    "Kennedy",
    "Fontibón",
    "Engativá",
    "Suba",
    "Barrios Unidos",
    "Teusaquillo",
    "Los Mártires",
    "Antonio Nariño",
    "Puente Aranda",
    "La Candelaria",
    "Rafael Uribe Uribe",
    "Ciudad Bolívar",
    "Sumapaz",
]

def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # strip accents
    s = s.lower().strip()

    # remove common OSM prefixes
    s = re.sub(r"^(localidad\s+de\s+|localidad\s+del\s+|localidad\s+)", "", s)

    # normalize punctuation/spaces
    s = s.replace(".", " ").replace(",", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Build normalized lookup: norm(name)->canonical accented name
wanted = {norm(x): x for x in LOCALIDADES}
wanted_keys = set(wanted.keys())

# Pull admin boundaries; this returns many admin things, we filter down
gdf = ox.features_from_place(PLACE, tags={"admin_level": True})

# Keep only polygons
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

# Find best name field (OSM can be weird)
name_col = None
for c in ["name", "official_name", "short_name", "alt_name"]:
    if c in gdf.columns:
        name_col = c
        break
if name_col is None:
    raise SystemExit("No name-like column found in OSM result.")

gdf[name_col] = gdf[name_col].fillna("").astype(str)
gdf["admin_level"] = gdf.get("admin_level", "").fillna("").astype(str)
gdf["name_norm"] = gdf[name_col].map(norm)

# For Bogotá localidades, admin_level=8 is usually the right one
gdf8 = gdf[gdf["admin_level"] == "8"].copy()

# Match against our wanted list
hits = gdf8[gdf8["name_norm"].isin(wanted_keys)].copy()

if hits.empty:
    # Debug dump: show what OSM thinks the names are at admin_level 8
    present = sorted(set(gdf8["name_norm"]))
    raise SystemExit(
        "No localidad matches after normalization.\n"
        f"Admin level 8 normalized names sample:\n{present[:60]}"
    )

# Canonical accented name
hits["name"] = hits["name_norm"].map(lambda k: wanted[k])

# One geometry per localidad (dissolve duplicates)
localidades = hits[["name", "geometry"]].dissolve(by="name", as_index=False)

# Sanity: check missing localidades
have = set(norm(x) for x in localidades["name"].tolist())
missing = sorted(wanted_keys - have)
if missing:
    print("WARNING: missing these localidades after match:", [wanted[m] for m in missing])

# Write full localidades file (20 features expected)
localidades.to_file("bogota_localidades.geojson", driver="GeoJSON")
print(f"Wrote bogota_localidades.geojson with {len(localidades)} features:")
print(sorted(localidades["name"].tolist()))

# Write AOI union file (1 feature) for clipping
aoi_geom = unary_union(localidades.geometry)
aoi = gpd.GeoDataFrame([{"name": "AOI_localidades", "geometry": aoi_geom}], crs=localidades.crs)
aoi.to_file("aoi_localidades.geojson", driver="GeoJSON")
print(f"Wrote aoi_localidades.geojson with 1 feature. Geometry type: {aoi.geometry.iloc[0].geom_type}")
