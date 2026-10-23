"""Chronos-2 cells of the suite on Modal H100s, one container per cell; results land in <run dir>/.

Run only after the user's Go. Data on the projectb-data volume: data/<box>/{built,era5}.npz (modal volume put);
the run's plan.npz is sent with each call. Weights are downloaded from Hugging Face inside the container.
  modal run revision/c2_modal.py --run runs/20261022_1639_suite --cells skill,drought,sens,world:96
"""
import modal

app = modal.App("projectb-c2")
vol = modal.Volume.from_name("projectb-data")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy", "chronos-forecasting==2.3.2", "torch==2.14.1", "transformers==5.20.2")
    .add_local_python_source("aoi", "run")
    .add_local_file("revision/c2.py", "/root/c2.py")
)


@app.function(image=image, gpu="H100", memory=65536, timeout=4 * 3600, volumes={"/vol": vol}, retries=0)
def cell(cmd: str, extra: list, plan: bytes):
    import os
    import subprocess
    import time
    os.chdir("/vol")  # c2 reads data/<box>/*.npz relative to the working directory
    out = "/tmp/out"
    os.makedirs(out, exist_ok=True)
    open(f"{out}/plan.npz", "wb").write(plan)
    t = time.time()
    subprocess.run(["python", "/root/c2.py", cmd, out, *extra], check=True)
    log = f"{cmd} {' '.join(extra)}: {time.time() - t:.0f} s on H100\n"
    open(f"{out}/gpu_seconds_{cmd}.txt", "w").write(log)
    return {f: open(f"{out}/{f}", "rb").read() for f in os.listdir(out) if f != "plan.npz"}


@app.local_entrypoint()
def main(run: str, cells: str):
    jobs = [(c.split(":")[0], ["--L", c.split(":")[1]] if ":" in c else []) for c in cells.split(",")]
    plan = open(f"{run}/plan.npz", "rb").read()
    for (c, _), files in zip(jobs, cell.starmap([(c, e, plan) for c, e in jobs], return_exceptions=True)):
        if isinstance(files, Exception):  # one failed cell does not lose the others
            print("FAILED", c, repr(files)[:500])
            continue
        for name, data in files.items():
            open(f"{run}/{name}", "wb").write(data)
            print("wrote", f"{run}/{name}")
