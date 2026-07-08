# CLAUDE.md — HE-Libaries-

Reproducible benchmark measuring the practical overhead of Homomorphic Encryption (CKKS via TenSEAL) vs AES/RSA baselines for simple numerical analytics: runtime, memory, ciphertext size, approximation error.

## Tech stack
- Python 3.11 only (TenSEAL 0.3.16 has no cp313 wheel; numpy pinned < 2.0 for ABI)
- tenseal 0.3.16, numpy 1.26.4, pandas 2.2.2, matplotlib 3.8.4, cryptography 42.0.8
- pytest 8.2.2, Streamlit 1.37.1 + plotly for the dashboard
- Docker is the canonical environment (`requirements.lock` installed in image)

## Commands
```
docker compose build                             # build image
docker compose run --rm benchmark pytest -v      # correctness tests
docker compose run --rm benchmark                # full benchmark → results/results.csv
docker compose up dashboard                      # Streamlit UI → http://localhost:8501
```
After changing `requirements.txt`, regenerate the lock:
`docker run --rm he-benchmark python -m pip freeze > requirements.lock`

## Architecture
- `config.py` — all tunable parameters, single source
- `he_benchmark/` — core: `he_ckks.py` (CKKS ops), `baseline_aes.py`/`baseline_rsa.py`, `metrics.py` (timing + peak RSS), `reference.py` (NumPy ground truth), `data.py` (seeded synthetic data)
- `app/streamlit_app.py` — dashboard over results.csv
- `results/` + `figures/` — benchmark outputs (csv, run_config.json, depth_sweep.json, PNGs)
- `report/` — FINAL_REPORT.md; build_docx.js renders the .docx

## Rules
- Reproducibility is the point: seeded data, pinned deps, run_config.json alongside results. Don't unpin versions or change seeds casually.
- Don't upgrade numpy to 2.x or Python past 3.11 without checking TenSEAL wheels.
- Committed results/figures back a written report — regenerating them changes the report's numbers. Regenerate only when asked.
