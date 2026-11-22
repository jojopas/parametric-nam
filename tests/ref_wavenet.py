"""A slow numpy WaveNet for tests: plays a plain or parametric .nam dict (SPEC.md), with
knob_scale, straight from the flat weights. Not used by the tools themselves."""
import numpy as np

ACT = {"Tanh": np.tanh, "LeakyReLU": lambda z: np.where(z > 0, z, 0.01 * z), "ReLU": lambda z: np.maximum(z, 0)}


def receptive_field(model):
    return 1 + sum((L["kernel_size"] - 1) * sum(L["dilations"]) for L in model["config"]["layers"])


def play(model, x, g=()):
    """x: audio (1-D). g: knob values (empty for a plain file). Returns the last len(x) - rf + 1 samples."""
    w = np.asarray(model["weights"], dtype=np.float64)
    ks = model.get("knob_scale")
    g = np.asarray(g, dtype=np.float64)
    N = g.size
    p = 0

    def take(n):
        nonlocal p
        s = w[p:p + n]
        p += n
        return s

    T0 = len(x)
    cond = np.vstack([np.asarray(x, dtype=np.float64)[None, :], np.repeat(g[:, None], T0, axis=1)]) if N else \
        np.asarray(x, dtype=np.float64)[None, :]
    y, head = cond, None
    for a, L in enumerate(model["config"]["layers"]):
        ch, K, hd, cin, cs = L["channels"], L["kernel_size"], L["head_size"], L["input_size"], L["condition_size"]
        assert cs == 1 + N and not L.get("gated", False)
        f = ACT[L["activation"]]
        xa = take(ch * cin).reshape(ch, cin) @ y
        for l, d in enumerate(L["dilations"]):
            W = take(ch * ch * K).reshape(ch, ch, K)
            b = take(ch)
            M = take(ch * cs).reshape(ch, cs)
            W1 = take(ch * ch).reshape(ch, ch)
            b1 = take(ch)
            T = xa.shape[1] - (K - 1) * d
            zc = sum(W[:, :, k] @ xa[:, k * d:k * d + T] for k in range(K)) + b[:, None]
            c = cond[:, -T:]
            if ks is not None:
                S = np.asarray(ks[a][l], dtype=np.float64).reshape(ch, N)
                s = 1.0 + S @ g
                z = (zc + M[:, 1:] @ c[1:]) * s[:, None] + M[:, :1] @ c[:1]
            else:
                z = zc + M @ c
            post = f(z)
            xa = xa[:, -T:] + W1 @ post + b1[:, None]
            head = post if head is None else head[:, -T:] + post
        Hw = take(hd * ch).reshape(hd, ch)
        Hb = take(hd) if L.get("head_bias", True) else np.zeros(hd)
        head = Hw @ head + Hb[:, None]
        y = xa
    hs = take(1)[0]
    assert p == w.size
    return hs * head[0]
