"""M3 fits on Modal H100s, in parallel; results land in <run dir>/gpu/.

Data on the projectb-data volume: data/<box>/{built,topo,era5}.npz (modal volume put).
  modal run revision/m3_modal.py --run runs/20261021_0147_m3 --fits topo:0,topo:1,topo_pet:0,topo_pet:1
"""
import modal

app = modal.App("projectb-m3")
vol = modal.Volume.from_name("projectb-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "optax", "jax[cuda12]")
    .add_local_python_source("aoi", "model", "run")
    .add_local_file("revision/m3.py", "/root/m3.py")
)


@app.function(image=image, gpu="H100", memory=65536, timeout=3600, volumes={"/vol": vol}, retries=0)
def fit(config: str, seed: int, m0_json: str):
    import os
    os.chdir("/vol")  # m3 reads data/<box>/*.npz relative to the working directory
    out = "/tmp/out"
    os.makedirs(out, exist_ok=True)
    open(f"{out}/metrics_M0.json", "w").write(m0_json)
    import m3
    m3.run_fit(config, seed, out)
    return {f: open(f"{out}/{f}", "rb").read() for f in os.listdir(out) if f != "metrics_M0.json"}


@app.local_entrypoint()
def main(run: str, fits: str):
    import os
    jobs = [(c, int(s)) for c, s in (f.split(":") for f in fits.split(","))]
    m0 = open(f"{run}/metrics_M0.json").read()
    os.makedirs(f"{run}/gpu", exist_ok=True)
    for files in fit.starmap([(c, s, m0) for c, s in jobs]):
        for name, data in files.items():
            open(f"{run}/gpu/{name}", "wb").write(data)
            print("wrote", f"{run}/gpu/{name}")
