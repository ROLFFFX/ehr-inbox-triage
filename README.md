## Structure

| Path | Contents |
|---|---|
| `triage/data.py` | Loaders for PMR-Bench (Reddit, Synth) and the C-SSRS r/SuicideWatch corpus, all in one schema |
| `triage/metrics.py` | Threshold locking for both operating points, frontier metrics, prevalence transport and workload (PPV, flags per week, reads per urgent message), stratified bootstrap, ranking metrics (T-NDCG, pairwise accuracy, urgent queue position) |
| `triage/methods/` | `rules.py` (lexicon, negation, 4 rule variants), `tfidf.py` (LR, ordinal, hybrid, keyword bypass, filter cascade), `encoder.py` (fine-tuned Bio_ClinicalBERT and RoBERTa, binary and ordinal heads) |
| `triage/harness.py` | The protocol: CV calibration → locked thresholds → classification track, ranking track, psychiatry transfer |
| `scripts/` | Download, run (`run_baselines`, `run_encoders`), figure, table and rule-analysis entry points |
| `notebooks/encoders_colab.ipynb` | Runs the encoder arm on a Colab GPU and saves results to Google Drive |
| `results/` | Result CSVs in `primary/` (Level 1 = urgent) and `sens_level2/` (Levels 1–2); `figures/` |
| `data/raw/` | Downloaded corpora (created by `scripts.download_data`, not committed) |
| `tests/` | Unit tests for metrics and rules |

## Setup

- **Python 3.11 or newer** (developed on 3.13, macOS).
- **Baselines** need only the CPU and `requirements.txt`.
- **Encoders** need PyTorch and `requirements-gpu.txt`. They were run on a Colab T4 GPU. Apple-silicon MPS works for smoke tests but is too slow for full runs.
- **Data** comes from the Hugging Face Hub with no token or login: `scripts.download_data` saves about 5 MB to `data/raw/`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch -r requirements-gpu.txt
.venv/bin/python -m scripts.download_data
```

## Reproduce

```bash
.venv/bin/python -m scripts.run_baselines
.venv/bin/python -m scripts.run_baselines --max-level 2 --tag sens_level2
.venv/bin/python -m scripts.analyze_rules
.venv/bin/python -m scripts.make_figures && .venv/bin/python -m scripts.make_tables
.venv/bin/python -m pytest -q tests
```

`make_tables` writes LaTeX tables to `report/tables/`, which is created if it does not exist. The report itself is not part of this repository.

The encoders need a GPU and the extra packages in `requirements-gpu.txt` (plus PyTorch, which Colab already has). The simplest route is to open `notebooks/encoders_colab.ipynb` in Colab and run all cells. To run them locally instead:

```bash
.venv/bin/python -m scripts.run_encoders
.venv/bin/python -m scripts.run_encoders --model-seed 1 --tag seeds/s1
```

Encoder rows are appended to the existing result files, so run the baselines first.