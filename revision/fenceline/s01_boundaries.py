"""Step 01: fetch property polygons from OpenStreetMap (Overpass API) for a box around Box B.
Saves the raw response unmodified, then a candidate list of named area features. No selection
of treated/control happens here; that is fixed in the exposure coding (step 03) and PREREG."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import fl_common as C
import requests

C.start("01_boundaries", __file__)
S, W, Nn, E = 0.15, 36.60, 0.75, 37.20   # Box B plus ~0.1 deg margin so straddling polygons are complete
q = f"""[out:json][timeout:180][bbox:{S},{W},{Nn},{E}];
(
  way["boundary"]["boundary"!="administrative"]["name"]; relation["boundary"]["boundary"!="administrative"]["name"];
  way["leisure"="nature_reserve"]; relation["leisure"="nature_reserve"];
  way["landuse"]["name"]; relation["landuse"]["name"];
);
out geom;"""
r = None
for url in ("https://overpass.kumi.systems/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://overpass-api.de/api/interpreter"):
    t0 = time.time()
    try:
        r = requests.post(url, data={"data": q}, timeout=240, headers={"User-Agent": "ProjectB-fenceline-research"})
        r.raise_for_status(); r.json()
        break
    except Exception as ex:
        C.log("01_boundaries", "mirror failed", url=url, error=repr(ex)[:300]); r = None
assert r is not None, "all Overpass mirrors failed"
raw = C.out("inputs", "osm", "overpass_raw.json")
h = C.write_new(raw, r.text)
C.log("01_boundaries", "Overpass query saved", url=url, query=q.replace("\n", " "), bytes=len(r.text),
      seconds=round(time.time() - t0, 1), file=os.path.relpath(raw, C.ROOT), sha256=h,
      osm_timestamp=r.json().get("osm3s", {}).get("timestamp_osm_base"))
els = r.json()["elements"]
rows = []
for e in els:
    t = e.get("tags", {})
    if "name" not in t:
        continue
    if t.get("boundary") == "administrative":
        continue
    rows.append({"type": e["type"], "id": e["id"], "name": t.get("name"),
                 "tags": {k: v for k, v in t.items() if k in ("boundary", "leisure", "landuse", "place", "protect_class",
                                                            "protection_title", "operator", "ownership", "website")}})
rows.sort(key=lambda z: z["name"])
cand = C.out("inputs", "osm", "named_area_features.json")
C.write_new(cand, json.dumps(rows, indent=1))
C.log("01_boundaries", "named non-administrative area features listed", n=len(rows),
      names="; ".join(sorted({z["name"] for z in rows})))
