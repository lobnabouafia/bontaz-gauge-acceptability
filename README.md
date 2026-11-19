# From Ranking Quality to Gauge Acceptability

Code and data for the paper:

> Bouafia L, Achouri A, El Moubarki L. *From Ranking Quality to Gauge Acceptability: An AIAG Attribute Gauge Study of an Unsupervised Visual Inspection Model.* Submitted to *Quality and Reliability Engineering International*.

The study judges a learned visual inspection model as a gauge — false-alarm rate α, miss rate β, effectiveness E and agreement with a reference and with human appraisers (AIAG attribute agreement analysis) — instead of by the AUROC alone.

## What is shared and what is not

The photographs belong to BONTAZ Tunisie and are **not** included. Everything needed to recompute the reported numbers from the model outputs is included:

| File | Content |
|---|---|
| `data/photo_list.csv` | Every development and gauge-study photograph: anonymous identifier, set (training, calibration, validation, test, gauge study), reference decision (OK/NOK) and part number for gauge-study parts |
| `data/model_scores.csv` | Score of each photograph for the two detectors (`recipe` = the reported model, `standard` = standard PatchCore) on the held-out test, the gauge set and the matched set, with the distance to the nearest training photograph (`nn`) and the part/background region scores for three disc sizes (`in*`, `out*`, `peak*`) |
| `data/operator_decisions.csv` | The 450 operator decisions of the AIAG study (3 operators × 3 trials × 50 parts) with the reference and the part difficulty (clear/borderline) |
| `data/public_controls.csv` | Results of the public controls on MVTec AD (metal nut, screw; clean and with eight artificial acquisition shortcuts; three splits) |
| `data/diagnostics_report.txt` | Printed report of the diagnostics run (both detectors, public controls) |

## Reproduce the numbers and figures (CPU, seconds)

```bash
pip install -r requirements.txt
python analysis/paper_numbers.py     # Tables 3-4, Sections 5.1-5.4
python analysis/make_figures.py      # Figures 1-4 (Figure 5 needs the photographs)
```

## Re-run the models (needs the photographs; Google Colab)

| Script | What it does |
|---|---|
| `src/01_fit_and_score_model.py` | Fits the reported model on the training photographs, fixes the threshold on the calibration photographs (96th percentile) and scores all sets |
| `src/02_diagnostics_region_scales.py` | Closeness to the training data, part/background split at three disc sizes, anomaly maps |
| `src/03_standard_patchcore_and_public_controls.py` | Standard PatchCore on the same data and the MVTec AD public controls |

The public controls can be re-run by anyone with MVTec AD. The BONTAZ scripts need the confidential photographs.

## License

MIT (code). The data files may be used for verification and research with citation of the paper.
