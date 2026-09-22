"""Run the full pipeline on a Modal GPU.

  modal volume put projectb-data data/A/built.npz data/A/built.npz
  modal volume put projectb-data data/B/built.npz data/B/built.npz
  modal deploy train.py && python -c "import modal; print(modal.Function.from_name('projectb-train','train').spawn((800,800,2000)).object_id)"
  # progress: modal volume get projectb-data runs/latest/progress.log - ; results: modal volume get projectb-data runs/latest ./runs/<dir>
"""
import modal

app = modal.App("projectb-train")
vol = modal.Volume.from_name("projectb-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "optax", "jax[cuda12]")
    .add_local_python_source("aoi", "model", "run")
)


@app.function(image=image, gpu="H100", memory=65536, timeout=3600 * 3, volumes={"/vol": vol}, retries=0)
def train(steps: tuple):
    import io
    import os
    import numpy as np
    import run

    out = "/vol/runs/latest"
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(f"{out}/{f}")
    n = [0]

    def log(msg):
        print(msg, flush=True)
        with open(f"{out}/progress.log", "a") as fh:
            fh.write(msg + "\n")
        n[0] += 1
        if n[0] % 10 == 0 or not msg.startswith("  step"):
            vol.commit()

    metrics, arrays, params = run.pipeline(data_dir="/vol/data", out=out, steps=steps, log=log)
    vol.commit()
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return metrics, buf.getvalue(), {k: {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in params.items()}


@app.local_entrypoint()
def main(steps: str = "800,800,2000"):
    import json
    import numpy as np
    import run

    out = run.run_dir("pipeline_gpu")
    metrics, arrays_bytes, params = train.remote(tuple(int(s) for s in steps.split(",")))
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    with open(f"{out}/arrays.npz", "wb") as f:
        f.write(arrays_bytes)
    np.save(f"{out}/params.npy", params, allow_pickle=True)
    print("wrote", out)
