"""Step 01c: which OSM polygons overlap Box B, and by how many pixels (pixel-centre rasterisation).
Geometry only; feeds the tenure/exposure coding."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import fl_common as C
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin

C.start("01c_overlap", __file__)
TR = from_origin(C.BOX_B[0], C.BOX_B[3], C.RES, C.RES)
fc = json.load(open(C.out("inputs", "osm", "polygons.geojson")))
rows = []
for f in fc["features"]:
    m = rasterize([(f["geometry"], 1)], out_shape=(C.N, C.N), transform=TR, fill=0, dtype="uint8")
    if m.sum():
        rows.append({"osm": f["properties"]["osm"], "name": f["properties"]["name"],
                     "category": f"{f['properties']['category']}/{f['properties']['type']}", "pixels_in_box": int(m.sum())})
rows.sort(key=lambda r: -r["pixels_in_box"])
p = C.out("inputs", "osm", "overlap_box_B.json")
C.write_new(p, json.dumps(rows, indent=1))
C.log("01c_overlap", "polygons overlapping Box B", file=os.path.relpath(p, C.ROOT), sha256=C.sha256(p),
      overlap="; ".join(f"{r['name']} ({r['osm']}) {r['pixels_in_box']} px" for r in rows))
