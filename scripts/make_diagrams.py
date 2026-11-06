"""Diagram konseptual Proposal UTS (300 dpi): CRISP-DM, pipeline bebas leakage, arsitektur deployment."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
INK, INK2, LINE = "#0b0b0b", "#52514e", "#8a8983"
BLUE, BLUE_BG = "#1f4e9c", "#e8f0fb"
ORANGE, ORANGE_BG = "#b4531f", "#fdf0e8"
GREEN, GREEN_BG = "#13795b", "#e6f4ee"
GREY_BG = "#f4f4f2"
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300})


def box(ax, x, y, w, h, title, body, edge, bg, tsize=9.5, bsize=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=bg, ec=edge, lw=1.3))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color=edge)
    ax.text(x + w / 2, y + h - 0.062, body, ha="center", va="top", fontsize=bsize, color=INK, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=LINE, style="-|>", rad=0.0, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


# ------------------------------------------------------------------ Gambar 3.1 CRISP-DM
d, sp, th = R["data"], R["split"], R["threshold"]
ft = R["final_test"]["thr_sel"]
fig, ax = plt.subplots(figsize=(11, 6.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
W, H = 0.29, 0.34
pos = [(0.025, 0.58), (0.355, 0.58), (0.685, 0.58), (0.685, 0.12), (0.355, 0.12), (0.025, 0.12)]
items = [
    ("1. Business Understanding", "Kargo ekspor terlambat bila berangkat\npabrik > cut-off gudang forwarder\n"
     f"({d['pct_delayed']}% dari {d['n_labeled']:,} kargo).\nTitik prediksi: saat booking & cut-off\n"
     "terjadwal. Biaya FN ≫ FP (gagal kapal,\nrebooking, demurrage).\n[Tugas 01]"),
    ("2. Data Understanding", f"Ekspor ERP NTIS {d['n_raw']:,} × {d['n_cols_raw']} kolom.\n"
     f"{d['n_labeled']:,} berlabel final, {d['n_pending']} pending.\nEDA: rasio 7,38 : 1; QTYO right-skewed;\n"
     f"{d['n_style']} STYLE ({R['eda']['style_freq_le5']} muncul ≤ 5 kali).\nAudit leakage: Buffer = label;\n"
     f"Contrac date > berangkat ({R['leakage_audit']['contract_after_departure_share']*100:.1f}%).\n[Tugas 02–04]"),
    ("3. Data Preparation", "7 fitur pra-keberangkatan (Order_Month\ndikeluarkan, hasil ablasi).\n"
     "imblearn Pipeline per fold: Imputer →\n(SMOTENC) → RobustScaler + One-Hot\n+ TargetEncoder (cross-fitting).\n"
     f"Split stratified 80:20 ({sp['n_train']:,} / {sp['n_test']}).\n[Tugas 02–03]"),
    ("4. Modeling", "Stratified 5-Fold CV untuk semua seleksi.\nBaseline: LR, Decision Tree.\n"
     "Ensemble: Random Forest, LightGBM,\nXGBoost (parameter default).\nOptuna TPE: 4 studi × 100 trial\n"
     "(XGB/LGBM × class weight/SMOTENC).\n[Tugas 03–04, 06]"),
    ("5. Evaluation", f"Data uji dipakai sekali. ROC-AUC {ft['roc_auc']:.3f},\nPR-AUC {ft['pr_auc']:.3f}; bootstrap 95% CI.\n"
     f"Threshold dari OOF: biaya FN:FP = 5:1\n→ {th['selected']:.2f} (≈ kapasitas 20% → {th['capacity20']:.2f}).\n"
     "Confusion matrix, ROC/PR, TreeSHAP,\nvalidasi kronologis.\n[Tugas 04, 06]"),
    ("6. Deployment Planning", "Pipeline utuh → model_final_uts.joblib\n(data mentah ERP → probabilitas).\n"
     "Prototipe Flask + REST API (Tugas 05);\nUAS: antarmuka Streamlit, threshold\n"
     f"{th['selected']:.2f}, zona review & prioritas,\nmonitoring & retraining bulanan.\n[Tugas 05 → UAS]"),
]
cols = [(BLUE, BLUE_BG)] * 3 + [(ORANGE, ORANGE_BG)] * 2 + [(GREEN, GREEN_BG)]
for (x, y), (t, b), (e, bg) in zip(pos, items, cols):
    box(ax, x, y, W, H, t, b, e, bg, bsize=7.7)
for (x1, y1), (x2, y2) in [((0.315, 0.75), (0.355, 0.75)), ((0.645, 0.75), (0.685, 0.75)),
                           ((0.83, 0.58), (0.83, 0.46)), ((0.685, 0.29), (0.645, 0.29)), ((0.355, 0.29), (0.315, 0.29))]:
    arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.6)
arrow(ax, 0.17, 0.46, 0.17, 0.58, color=INK2, lw=1.2, ls=(0, (4, 3)))
ax.text(0.18, 0.52, "umpan balik label & retraining", fontsize=7.3, color=INK2, va="center")
arrow(ax, 0.50, 0.46, 0.50, 0.58, color=ORANGE, lw=1.2, ls=(0, (4, 3)))
ax.text(0.51, 0.52, "iterasi: ablasi fitur, ganti strategi imbalance", fontsize=7.3, color=ORANGE, va="center")
ax.text(0.5, 0.035, "Biru = pemahaman & penyiapan data · Oranye = pemodelan & evaluasi · Hijau = rencana deployment (UAS)",
        ha="center", fontsize=7.6, color=INK2)
fig.savefig(f"{OUT}/gambar3_1_crispdm.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ------------------------------------------------------------------ Gambar 3.3 Pipeline bebas leakage
fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.01, 0.62, 0.17, 0.32, "Data berlabel", f"{d['n_labeled']:,} kargo\n7 fitur + target\n(Delayed {d['pct_delayed']}%)\n\nsplit stratified 80:20\nrandom_state = 42", INK2, GREY_BG)
box(ax, 0.24, 0.74, 0.17, 0.20, "Data latih 80%", f"{sp['n_train']:,} kargo\n({sp['train_delayed']} Delayed)", BLUE, BLUE_BG)
box(ax, 0.24, 0.42, 0.17, 0.22, "Data uji 20%", f"{sp['n_test']} kargo ({sp['test_delayed']} Delayed)\nDIKUNCI — tidak disentuh\nsaat tuning & threshold", ORANGE, ORANGE_BG)
arrow(ax, 0.18, 0.84, 0.24, 0.84); arrow(ax, 0.18, 0.68, 0.24, 0.56)
ax.add_patch(FancyBboxPatch((0.45, 0.33), 0.535, 0.63, boxstyle="round,pad=0.006,rounding_size=0.012",
                            fc="white", ec=BLUE, lw=1.2, ls=(0, (5, 3))))
ax.text(0.717, 0.945, "Stratified 5-Fold CV pada data latih — diulang untuk setiap fold k = 1…5", ha="center",
        va="top", fontsize=8.6, fontweight="bold", color=BLUE)
box(ax, 0.47, 0.70, 0.13, 0.17, "4 fold latih", "≈ 2.011 kargo\n(240 Delayed)", BLUE, BLUE_BG, bsize=7.2)
box(ax, 0.47, 0.43, 0.13, 0.17, "1 fold validasi", "≈ 503 kargo\n(60 Delayed)", GREEN, GREEN_BG, bsize=7.2)
steps = [("Imputer", "median /\nmodus"), ("SMOTENC*", "sintetis kelas\nminoritas"),
         ("Encoder &\nScaler", "RobustScaler\nOne-Hot\nTargetEncoder"), ("Model", "XGBoost /\nLightGBM")]
xs = [0.625, 0.715, 0.805, 0.895]
for x, (t, b_) in zip(xs, steps):
    ax.add_patch(FancyBboxPatch((x, 0.68), 0.08, 0.21, boxstyle="round,pad=0.004,rounding_size=0.01",
                                fc=BLUE_BG, ec=BLUE, lw=1.1))
    ax.text(x + 0.04, 0.875, t, ha="center", va="top", fontsize=7.4, fontweight="bold", color=BLUE)
    ax.text(x + 0.04, 0.79, b_, ha="center", va="top", fontsize=6.6, color=INK)
for x in xs[:-1]:
    arrow(ax, x + 0.08, 0.785, x + 0.09, 0.785, color=BLUE, lw=1.1)
arrow(ax, 0.60, 0.785, 0.625, 0.785, color=BLUE, lw=1.1)
ax.text(0.80, 0.652, "fit()  hanya pada fold latih", ha="center", fontsize=7.5, color=BLUE, fontweight="bold")
ax.add_patch(FancyBboxPatch((0.625, 0.43), 0.35, 0.17, boxstyle="round,pad=0.004,rounding_size=0.01",
                            fc=GREEN_BG, ec=GREEN, lw=1.1))
ax.text(0.80, 0.585, "transform() + predict_proba() memakai statistik fold latih", ha="center", va="top",
        fontsize=7.5, fontweight="bold", color=GREEN)
ax.text(0.80, 0.54, "SMOTENC dilewati (tidak ada data sintetis di validasi)\nskor fold: PR-AUC, ROC-AUC, F1-Macro, recall\n+ prediksi OOF untuk memilih threshold",
        ha="center", va="top", fontsize=7, color=INK, linespacing=1.35)
arrow(ax, 0.60, 0.515, 0.625, 0.515, color=GREEN, lw=1.1)
arrow(ax, 0.41, 0.84, 0.47, 0.80, color=BLUE)
ax.text(0.717, 0.41, "Optuna mengulang skema ini pada setiap trial (4 studi × 100 trial)\n* SMOTENC hanya pada skenario SMOTE",
        ha="center", va="top", fontsize=7.0, color=INK2, linespacing=1.3)
box(ax, 0.50, 0.06, 0.485, 0.20, "Model final + threshold dikunci",
    f"{R['final_model']} di-fit ulang pada seluruh data latih;\nthreshold {th['selected']:.2f} dari prediksi OOF (biaya FN:FP = 5:1)", INK2, GREY_BG, bsize=7.4)
box(ax, 0.24, 0.06, 0.20, 0.20, "Evaluasi akhir (sekali)",
    f"ROC-AUC {ft['roc_auc']:.3f} · PR-AUC {ft['pr_auc']:.3f}\nrecall {ft['recall']:.3f} · FN {ft['FN']} dari {sp['test_delayed']}", ORANGE, ORANGE_BG, bsize=7.2)
arrow(ax, 0.74, 0.33, 0.74, 0.26, color=INK2)
arrow(ax, 0.50, 0.16, 0.44, 0.16, color=ORANGE)
arrow(ax, 0.325, 0.42, 0.325, 0.26, color=ORANGE)
fig.savefig(f"{OUT}/gambar3_3_pipeline.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ------------------------------------------------------------------ Gambar 3.9 Deployment
fig, ax = plt.subplots(figsize=(11, 4.9)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.01, 0.50, 0.17, 0.36, "ERP NTIS", "data order:\nSTYLE, QTYO, Dest,\nSmode, FWD, Season,\nContrac date,\ncut-off gudang", INK2, GREY_BG, bsize=7.3)
box(ax, 0.23, 0.50, 0.21, 0.36, "Antarmuka (UAS)", "Streamlit: form input &\nunggah CSV mingguan\n(prototipe Flask + REST\nAPI sudah ada, Tugas 05)", BLUE, BLUE_BG, bsize=7.3)
box(ax, 0.49, 0.50, 0.23, 0.36, "Inferensi terkalibrasi", "model_final_uts.joblib\n(pipeline utuh)\n→ P(Delayed)\n"
    f"threshold {th['selected']:.2f}\nTreeSHAP per kargo", ORANGE, ORANGE_BG, bsize=7.3)
box(ax, 0.77, 0.50, 0.22, 0.36, "Keputusan PPIC", f"< {th['selected']:.2f}: proses normal\n{th['selected']:.2f}–{th['priority_from']:.2f}: review manual\n"
    f"≥ {th['priority_from']:.2f}: prioritas aksi\n(lembur, pindah lini,\nrebooking lebih awal)", GREEN, GREEN_BG, bsize=7.3)
for x1, x2 in [(0.18, 0.23), (0.44, 0.49), (0.72, 0.77)]:
    arrow(ax, x1, 0.68, x2, 0.68, color=INK2, lw=1.6)
box(ax, 0.23, 0.10, 0.49, 0.22, "Monitoring & pembelajaran berkelanjutan",
    "Catat prediksi, keputusan reviewer, dan label aktual per PO → pantau recall, precision,\n"
    "beban review, dan drift distribusi (STYLE baru, rute baru) setiap bulan →\nretraining bulanan dengan validasi kronologis setelah label pending final",
    INK2, GREY_BG, bsize=7.3)
arrow(ax, 0.88, 0.50, 0.72, 0.25, color=INK2, rad=-0.2, ls=(0, (4, 3)))
arrow(ax, 0.23, 0.21, 0.10, 0.50, color=INK2, rad=-0.2, ls=(0, (4, 3)))
fig.savefig(f"{OUT}/gambar3_9_deployment.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("ok")
