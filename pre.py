import json, folium

g = json.load(open("buffered.geojson"))
geom = g["features"][0]["geometry"]

m = folium.Map(location=[4.58, -74.07], zoom_start=11, tiles="CartoDB positron")
folium.GeoJson(geom, name="buffer").add_to(m)
m.save("buffer_preview.html")
print("Wrote buffer_preview.html")
