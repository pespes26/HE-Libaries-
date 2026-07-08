"""Measure how CKKS error grows with multiplicative depth — and where it stops.

The main benchmark stays at depth <= 1 by design. The report's depth-limit claim
(the [60, 40, 40, 60] chain supports ~1-2 sequential ciphertext multiplications)
is verified *empirically* here instead of being asserted from parameters:

    depth 1:  x^2 = enc(x) * enc(x)          (consumes rescale level 1)
    depth 2:  x^4 = x^2 * x^2                (consumes rescale level 2)
    depth 3:  x^8 = x^4 * x^4                (no levels left -> must fail)

Squaring keeps both operands at the same modulus level, so each step is exactly
one additional sequential multiplication. Inputs are drawn from [1, 2) so the
relative error isolates CKKS noise growth from magnitude effects (x^8 < 256).

Writes results/depth_sweep.json; analysis/make_plots.py renders it as
figures/ckks_error_vs_depth.png. Runs in seconds:

    docker compose run --rm benchmark python experiments/depth_sweep.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import tenseal as ts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from he_benchmark import he_ckks  # noqa: E402

OUT_JSON = os.path.join(config.RESULTS_DIR, "depth_sweep.json")
MAX_DEPTH = 4  # one past the expected failure point, to prove the wall is real
N_VALUES = config.CKKS_SLOTS  # one full ciphertext


def errors(ref: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    abs_err = np.abs(got - ref)
    return float(abs_err.max()), float(np.mean(abs_err / np.abs(ref)))


def main() -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_SEED)
    x = rng.uniform(1.0, 2.0, size=N_VALUES)

    print("Building CKKS context ...")
    ctx = he_ckks.make_context()
    enc = ts.ckks_vector(ctx, x.tolist())
    ref = x.copy()

    levels = []
    print(f"{'depth':>5} {'expr':>6} {'max_abs_error':>15} {'mean_rel_error':>15}  status")
    for depth in range(1, MAX_DEPTH + 1):
        try:
            enc = enc * enc          # ct x ct squaring: exactly one more level
            ref = ref * ref
            got = np.asarray(enc.decrypt(), dtype=np.float64)
            max_abs, mean_rel = errors(ref, got)
            levels.append({"depth": depth, "expr": f"x^{2 ** depth}", "ok": True,
                           "max_abs_error": max_abs, "mean_rel_error": mean_rel})
            print(f"{depth:>5} {'x^' + str(2 ** depth):>6} {max_abs:>15.3e} "
                  f"{mean_rel:>15.3e}  ok")
        except Exception as e:  # modulus chain exhausted -> TenSEAL/SEAL refuses
            levels.append({"depth": depth, "expr": f"x^{2 ** depth}", "ok": False,
                           "error": f"{type(e).__name__}: {e}"})
            print(f"{depth:>5} {'x^' + str(2 ** depth):>6} {'-':>15} {'-':>15}  "
                  f"FAILED ({type(e).__name__}: {e})")
            break

    payload = {
        "description": "CKKS error vs multiplicative depth (repeated ct x ct squaring "
                       "of values in [1,2); failure = modulus chain exhausted)",
        "seed": config.RANDOM_SEED,
        "n_values": N_VALUES,
        "input_range": [1.0, 2.0],
        "ckks": {
            "poly_modulus_degree": config.CKKS_POLY_MODULUS_DEGREE,
            "coeff_mod_bit_sizes": config.CKKS_COEFF_MOD_BIT_SIZES,
            "global_scale": float(config.CKKS_GLOBAL_SCALE),
        },
        "levels": levels,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
