"""One learned transition density in place of a closed form.

State per pixel: anomaly a_t and antecedent-rain index w_t (a one-parameter bucket,
the land's memory of rain). The transition is a discrete-time drift-diffusion:

    w_{t+1} = lam(x) * w_t + r_{t+1}
    a_{t+1} = a_t + mu(a_t, n_t, w_t, r_{t+1}, phase, x) + sigma(...) * eps

where n_t is the 5x5 neighbourhood mean of a_t. mu and sigma are linear (M1) or
linear plus a zero-initialised MLP residual (M2). M0 is the driver-blind version.

Every product is a functional of this one object:
  rollout_mean  -> impact under a driver path, and Greeks by jvp
  rollout_sample -> impact distribution
  nll           -> anomaly score
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

HIDDEN = 64


# ----------------------------------------------------------------------------- features
def neighbourhood(a):
    """5x5 mean of a [N,N] field (edge-padded)."""
    k = jnp.ones((5, 5)) / 25.0
    ap = jnp.pad(a, 2, mode="edge")
    return jax.lax.conv_general_dilated(ap[None, None], k[None, None], (1, 1), "VALID")[0, 0]


def features(a, n, w, r, phase, x, use_driver):
    """Per-pixel feature vector [N,N,F]."""
    N = a.shape[0]
    ph = jnp.broadcast_to(phase, (N, N, 2))
    cols = [a[..., None], n[..., None], ph, x]
    if use_driver:
        cols += [w[..., None], r[..., None]]
    else:
        cols += [jnp.zeros_like(a)[..., None], jnp.zeros_like(a)[..., None]]
    return jnp.concatenate(cols, -1)


N_FEAT = 1 + 1 + 2 + 3 + 2


# ----------------------------------------------------------------------------- params
def init_params(key, mlp, sigmoid=False):
    k1, k2 = jax.random.split(key)
    p = {
        "lin_mu": jnp.zeros(N_FEAT), "b_mu": jnp.zeros(()),
        "lin_ls": jnp.zeros(N_FEAT), "b_ls": jnp.log(0.05),
        "lam": jnp.zeros(4),  # sigmoid(lam . [1, x]) ; starts at 0.5
    }
    if sigmoid:
        # M1.5: hand-written S-curve of greenness change in effective water u = w + r:
        # s(x) * A * (sigmoid((u - theta)/tau) - sigmoid(-theta/tau)); threshold theta, width tau,
        # per-pixel scale s(x) = exp(c . x). Starts at A = 0, i.e. exactly at M1.
        p.update({"sig_A": jnp.zeros(()), "sig_theta": jnp.array(1.0), "sig_ltau": jnp.log(0.5),
                  "sig_c": jnp.zeros(3)})
    if mlp:
        p["W1"] = jax.random.normal(k1, (N_FEAT, HIDDEN)) / jnp.sqrt(N_FEAT)
        p["b1"] = jnp.zeros(HIDDEN)
        p["W2"] = jnp.zeros((HIDDEN, 2))  # zero-init residual: M2 starts exactly at M1
        p["b2"] = jnp.zeros(2)
    return p


def lam(p, x):
    return jax.nn.sigmoid(p["lam"][0] + x @ p["lam"][1:])


def drift_diffusion(p, f, z=None):
    """z: optional latent [...,2] per pixel, (level shift, rain gain): mu += z0 + z1 * (w + r)."""
    mu = f @ p["lin_mu"] + p["b_mu"]
    if z is not None:
        mu = mu + z[..., 0] + z[..., 1] * (f[..., 7] + f[..., 8])
    ls = f @ p["lin_ls"] + p["b_ls"]
    if "sig_A" in p:
        u = f[..., 7] + f[..., 8]  # w + r (feature layout: a, n, phase(2), x(3), w, r)
        tau = jax.nn.softplus(p["sig_ltau"]) + 1e-3
        g = jax.nn.sigmoid((u - p["sig_theta"]) / tau) - jax.nn.sigmoid(-p["sig_theta"] / tau)
        mu = mu + jnp.exp(f[..., 4:7] @ p["sig_c"]) * p["sig_A"] * g
    if "W1" in p:
        h = jax.nn.gelu(f @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        mu, ls = mu + h[..., 0], ls + h[..., 1]
    return mu, jnp.exp(jnp.clip(ls, -6, 1))


def step(p, a, w, r_next, phase_next, x, use_driver, z=None):
    """One transition: returns (mu_next, sigma_next, w_next)."""
    f = features(a, neighbourhood(a), w, r_next, phase_next, x, use_driver)
    mu, sig = drift_diffusion(p, f, z)
    return a + mu, sig, lam(p, x) * w + r_next


# ----------------------------------------------------------------------------- likelihood
def nll_sequence(p, D, use_driver):
    """Per-step, per-pixel Gaussian NLL over the whole sequence: [T-1,N,N].
    Teacher forcing; w_t carried by scan from w_0 = climatological steady state."""
    x, rc0 = D["x"], D["rclim"][0]
    w0 = rc0 / (1 - lam(p, x))

    def body(w, t):
        mu, sig, w_next = step(p, D["a"][t], w, D["r"][t + 1], D["phase"][t + 1], x, use_driver)
        z = (D["a"][t + 1] - mu) / sig
        return w_next, 0.5 * z ** 2 + jnp.log(sig) + 0.5 * jnp.log(2 * jnp.pi)

    _, nll = jax.lax.scan(body, w0, jnp.arange(D["a"].shape[0] - 1))
    return nll


def fit(D, mask_t, use_driver, mlp, steps=1500, lr=3e-3, seed=0, val_t=None, log=print, sigmoid=False):
    """Full-batch Adam on masked one-step NLL. mask_t: [T-1] bool of usable target steps.
    Returns params at the best validation NLL (if val_t given) else the last."""
    p = init_params(jax.random.PRNGKey(seed), mlp, sigmoid)
    obs = D["obs"][1:] & D["obs"][:-1]
    m_train = obs & mask_t[:, None, None]
    m_val = obs & val_t[:, None, None] if val_t is not None else None

    n_train = float(m_train.sum())  # python floats: keeps XLA from constant-folding the mask sum
    n_val = float(m_val.sum()) if m_val is not None else 1.0

    def loss(p):
        nll = nll_sequence(p, D, use_driver)
        return jnp.sum(nll * m_train) / n_train

    @jax.jit
    def val_metrics(p):
        nll = nll_sequence(p, D, use_driver)
        return jnp.sum(nll * m_val) / n_val

    opt = optax.adam(lr)
    st = opt.init(p)

    @jax.jit
    def upd(p, st):
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    best, best_p = np.inf, p
    for i in range(steps):
        p, st, l = upd(p, st)
        if i % 50 == 0 or i == steps - 1:
            v = float(val_metrics(p)) if m_val is not None else float("nan")
            log(f"  step {i:5d} train nll {float(l):.4f} val nll {v:.4f}")
            if m_val is not None and v < best:
                best, best_p = v, p
    return (best_p if m_val is not None else p), best


# ----------------------------------------------------------------------------- rollouts
def rollout_mean(p, a0, w0, r_path, phase_path, x, use_driver, z=None):
    """Deterministic mean rollout over H steps: returns a_path [H,N,N]."""
    def body(c, inp):
        a, w = c
        r, ph = inp
        mu, _, w = step(p, a, w, r, ph, x, use_driver, z)
        return (mu, w), mu

    _, path = jax.lax.scan(body, (a0, w0), (r_path, phase_path))
    return path


def rollout_sample(p, key, a0, w0, r_path, phase_path, x, use_driver, S=16, z=None):
    """S sampled rollouts: [S,H,N,N]."""
    def one(k):
        def body(c, inp):
            a, w, k = c
            r, ph = inp
            k, sub = jax.random.split(k)
            mu, sig, w = step(p, a, w, r, ph, x, use_driver, z)
            a = mu + sig * jax.random.normal(sub, mu.shape)
            return (a, w, k), a

        _, path = jax.lax.scan(body, (a0, w0, k), (r_path, phase_path))
        return path

    return jax.vmap(one)(jax.random.split(key, S))


def w_path(p, D, use_driver=True):
    """Antecedent-rain index for every step [T,N,N] (deterministic given rain)."""
    x = D["x"]
    l = lam(p, x)
    w0 = D["rclim"][0] / (1 - l)

    def body(w, r):
        w = l * w + r
        return w, w

    _, ws = jax.lax.scan(body, w0, D["r"][1:])
    return jnp.concatenate([w0[None], ws], 0)


# ----------------------------------------------------------------------------- Greeks
def greeks(p, a0, w0, r_path, phase_path, x, use_driver=True):
    """Delta and gamma of E[a_h] with respect to a uniform rain perturbation in the
    first window of the path; theta from the linear response to the initial anomaly.
    Returns dict of [H,N,N] arrays."""
    H = r_path.shape[0]
    e0 = jnp.zeros_like(r_path).at[0].set(1.0)  # +100 mm in window 1, everywhere
    f = lambda r: rollout_mean(p, a0, w0, r, phase_path, x, use_driver)
    path, delta = jax.jvp(f, (r_path,), (e0,))
    _, gamma = jax.jvp(lambda r: jax.jvp(f, (r,), (e0,))[1], (r_path,), (e0,))
    g = lambda a: rollout_mean(p, a, w0, r_path, phase_path, x, use_driver)
    _, theta = jax.jvp(g, (a0,), (jnp.ones_like(a0),))  # dE[a_h]/da_0: persistence of an anomaly
    return {"mean": path, "delta": delta, "gamma": gamma, "theta": theta}


# ----------------------------------------------------------------------------- scores
def crps_gauss(mu, sig, y):
    z = (y - mu) / sig
    Phi = 0.5 * (1 + jax.scipy.special.erf(z / jnp.sqrt(2)))
    phi = jnp.exp(-0.5 * z ** 2) / jnp.sqrt(2 * jnp.pi)
    return sig * (z * (2 * Phi - 1) + 2 * phi - 1 / jnp.sqrt(jnp.pi))


def crps_samples(samples, y):
    """samples [S,...], y [...]: unbiased-ish energy-form CRPS."""
    S = samples.shape[0]
    t1 = jnp.mean(jnp.abs(samples - y[None]), 0)
    t2 = jnp.mean(jnp.abs(samples[:, None] - samples[None, :]), (0, 1)) * S / (S - 1)
    return t1 - 0.5 * t2
