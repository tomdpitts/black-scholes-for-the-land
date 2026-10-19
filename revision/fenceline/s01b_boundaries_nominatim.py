"""Step 01b: fallback after Overpass failed (logged). Fetch OSM polygons by name through Nominatim,
bounded to a box around Box B, 1 request/second (Nominatim usage policy). Raw responses are saved
verbatim; features are de-duplicated by (osm_type, osm_id). No treated/control choice here."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import fl_common as C
import requests

C.start("01b_boundaries", __file__)
C.log("01b_boundaries", "method change before freeze: Overpass unavailable (3 servers, 504/timeouts); using Nominatim name search with polygon_geojson")
W, S, E, Nn = 36.55, 0.10, 37.25, 0.80
names = ["Mpala", "Ol Jogi", "Loisaba", "Suyian", "Sosian", "Segera", "El Karama", "Ol Lentille", "Naibunga",
         "Il Polei", "Koija", "Tiamamut", "Kijabe", "Musul", "Munishoi", "Morupusi", "Il Motiok", "Ol Malo",
         "Lekiji", "Mugie", "Ol Maisor", "Laikipia National Reserve", "Ol Pejeta", "Jessel", "Kisima",
         "Laikipia Nature Conservancy", "Mpala Research Centre", "Ewaso", "Makurian", "Ngare Ndare",
         "conservancy", "ranch", "group ranch", "nature reserve"]
hdr = {"User-Agent": "ProjectB-fenceline-research (contact via project owner)"}
feats, raw_all = {}, {}
for q in names:
    params = {"q": q, "format": "geojson", "polygon_geojson": 1, "viewbox": f"{W},{Nn},{E},{S}", "bounded": 1,
              "limit": 40, "extratags": 1}
    r = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=hdr, timeout=60)
    time.sleep(1.1)
    r.raise_for_status()
    js = r.json(); raw_all[q] = js
    for f in js.get("features", []):
        pr = f["properties"]
        if f["geometry"]["type"] not in ("Polygon", "MultiPolygon"):
            continue
        key = f"{pr['osm_type']}/{pr['osm_id']}"
        feats[key] = {"type": "Feature", "geometry": f["geometry"],
                      "properties": {"osm": key, "name": pr.get("name") or pr.get("display_name", "").split(",")[0],
                                     "category": pr.get("category"), "type": pr.get("type"),
                                     "display_name": pr.get("display_name"), "extratags": pr.get("extratags"),
                                     "found_by_query": q}}
rawp = C.out("inputs", "osm", "nominatim_raw.json")
C.write_new(rawp, json.dumps(raw_all))
fc = {"type": "FeatureCollection", "features": sorted(feats.values(), key=lambda f: f["properties"]["name"] or "")}
fp = C.out("inputs", "osm", "polygons.geojson")
C.write_new(fp, json.dumps(fc))
C.log("01b_boundaries", "Nominatim polygons saved", queries=len(names), raw=os.path.relpath(rawp, C.ROOT),
      raw_sha256=C.sha256(rawp), file=os.path.relpath(fp, C.ROOT), sha256=C.sha256(fp), n_polygons=len(feats),
      polygons="; ".join(f"{k} {v['properties']['name']} [{v['properties']['category']}/{v['properties']['type']}]"
                         for k, v in sorted(feats.items(), key=lambda kv: kv[1]["properties"]["name"] or "")))
