# Prediksi Dini Keterlambatan Kargo Ekspor — PT Sun Suka Abadi (ERP NTIS)
Proposal UTS Pembelajaran Mesin 2026 · Kelompok 08 · Universitas Nusa Mandiri

Pipeline klasifikasi biner `Delivery Status` (1 = berangkat pabrik melewati waktu penutupan gudang) dengan
XGBoost + Optuna TPE + kalibrasi ambang batas berbasis biaya. Seluruh preprocessing (imputer, RobustScaler,
One-Hot, TargetEncoder cross-fitting, SMOTENC opsional) dibungkus `imblearn.pipeline.Pipeline` sehingga
di-fit ulang di setiap fold — bebas kebocoran data.

## Hasil utama (data uji n = 629, 75 Delayed; dipakai sekali)
| Model | Threshold | ROC-AUC | PR-AUC | Recall (1) | FN |
|---|---|---|---|---|---|
| XGBoost default | 0.50 | 0.8781 | 0.4892 | 0.293 | 53 |
| **XGBoost Optuna (class weight)** | **0.55** (biaya OOF 5:1, beban ≤ 20%) | **0.9042** | **0.5354** | **0.653** | **26** |

## Replikasi (Google Colab)
1. Buka `PROPOSAL_UTS_KELOMPOK_08.ipynb` di Google Colab.
2. `Runtime → Run all`. Sel 1 memasang pustaka sesuai `requirements.txt`.
3. Saat diminta, unggah `NTIS_DATASET.csv` (data internal; minta ke tim/dosen).
4. ±5 menit di CPU. Hasil identik (seed 42) tersimpan di `results/uts_results.json`, `results/optuna_trials_*.csv`,
   `figures/gambar3_*.png`, dan `model_final_uts.joblib`.

Replikasi lokal: `pip install -r requirements.txt`, lalu jalankan notebook di Jupyter dengan `NTIS_DATASET.csv`
di folder yang sama, atau `python scripts/build_and_run_nb.py scripts/nb_source.py out.ipynb .`.

## Struktur
| Berkas | Isi |
|---|---|
| `PROPOSAL_UTS_KELOMPOK_08.ipynb` | Notebook 20 bagian beranotasi, sudah dieksekusi |
| `ntis_prep.py` | Rekayasa fitur & ColumnTransformer (ditulis ulang otomatis oleh notebook) |
| `results/` | Seluruh angka naskah + riwayat 4 × 100 trial Optuna |
| `figures/` | Gambar 3.1–3.9 (300 dpi) |
| `scripts/` | Sumber notebook, runner, pembuat diagram & persamaan |

## Data & kerahasiaan
`NTIS_DATASET.csv` dan `model_final_uts.joblib` tidak disertakan (lihat `.gitignore`): keduanya memuat data
internal PT Sun Suka Abadi. Model dapat dibangkitkan ulang dari notebook.
