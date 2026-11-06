"""Render persamaan Bab II–III sebagai PNG 300 dpi (matplotlib mathtext, tanpa LaTeX)."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})

EQ = {
    "eq_sigmoid": r"$P(y=1\mid \mathbf{x}) = \sigma(\mathbf{w}^{\top}\mathbf{x}+b) = \dfrac{1}{1+e^{-(\mathbf{w}^{\top}\mathbf{x}+b)}}$",
    "eq_gini": r"$\mathrm{Gini}(S) = 1-\sum_{c\in\{0,1\}} p_c^{2},\qquad \Delta = \mathrm{Gini}(S)-\dfrac{|S_L|}{|S|}\mathrm{Gini}(S_L)-\dfrac{|S_R|}{|S|}\mathrm{Gini}(S_R)$",
    "eq_rf": r"$\hat{p}_{\mathrm{RF}}(\mathbf{x}) = \dfrac{1}{B}\sum_{b=1}^{B} \hat{p}_{b}(\mathbf{x}),\qquad \hat{p}_{b}\ \mathrm{dilatih\ pada\ sampel\ bootstrap\ ke}\text{-}b$",
    "eq_boost": r"$F_{m}(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta\, f_{m}(\mathbf{x}),\qquad \hat{p}(\mathbf{x}) = \sigma\!\left(F_{M}(\mathbf{x})\right)$",
    "eq_xgb": r"$\mathcal{L} = \sum_{i=1}^{n} w_{y_i}\,\ell\!\left(y_i,\hat{y}_i\right) + \sum_{m=1}^{M}\Omega(f_m),\qquad \Omega(f) = \gamma T + \dfrac{1}{2}\lambda\Vert\mathbf{w}\Vert_{2}^{2} + \alpha\Vert\mathbf{w}\Vert_{1}$",
    "eq_gain": r"$\mathrm{Gain} = \dfrac{1}{2}\left[\dfrac{G_L^{2}}{H_L+\lambda}+\dfrac{G_R^{2}}{H_R+\lambda}-\dfrac{(G_L+G_R)^{2}}{H_L+H_R+\lambda}\right]-\gamma$",
    "eq_robust": r"$\tilde{x} = \dfrac{x-\mathrm{median}(x)}{Q_{3}(x)-Q_{1}(x)}$",
    "eq_te": r"$\mathrm{TE}(k) = \dfrac{n_k\,\bar{y}_k + m\,\bar{y}}{n_k + m}$",
    "eq_smote": r"$\mathbf{x}_{\mathrm{baru}} = \mathbf{x}_i + \delta\,(\mathbf{x}_{nn}-\mathbf{x}_i),\qquad \delta\sim U(0,1)$",
    "eq_cm": r"$\mathrm{Accuracy} = \dfrac{TP+TN}{TP+TN+FP+FN},\qquad \mathrm{Precision} = \dfrac{TP}{TP+FP},\qquad \mathrm{Recall} = \dfrac{TP}{TP+FN}$",
    "eq_f1": r"$F_{1} = 2\cdot\dfrac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}},\qquad F_{1}^{\mathrm{macro}} = \dfrac{1}{2}\sum_{c\in\{0,1\}}F_{1,c},\qquad F_{1}^{\mathrm{weighted}} = \sum_{c}\dfrac{n_c}{n}F_{1,c}$",
    "eq_roc": r"$\mathrm{TPR}(t) = \dfrac{TP(t)}{TP(t)+FN(t)},\quad \mathrm{FPR}(t) = \dfrac{FP(t)}{FP(t)+TN(t)},\quad \mathrm{AUC} = \int_{0}^{1}\mathrm{TPR}\;d(\mathrm{FPR}) = P\!\left(\hat{p}(\mathbf{x}^{+})>\hat{p}(\mathbf{x}^{-})\right)$",
    "eq_ap": r"$\mathrm{AP} = \sum_{k}\left(R_{k}-R_{k-1}\right)P_{k}$",
    "eq_cost": r"$\hat{y} = \mathbb{1}\left[\hat{p}(\mathbf{x})\geq t\right],\qquad t^{*} = \arg\min_{t}\; C(t) = c_{FN}\cdot FN(t) + c_{FP}\cdot FP(t)$",
    "eq_tpe": r"$p(\theta\mid y) = \begin{cases} \ell(\theta) & y < y^{*}\\ g(\theta) & y \geq y^{*}\end{cases}\qquad \theta^{\mathrm{next}} = \arg\max_{\theta}\;\dfrac{\ell(\theta)}{g(\theta)}$",
}
# varian aman: matplotlib mathtext tidak mendukung \begin{cases} dan \text
EQ["eq_tpe"] = r"$\ell(\theta)=p(\theta\mid y\ \mathrm{terbaik}\ \gamma\%),\quad g(\theta)=p(\theta\mid y\ \mathrm{lainnya}),\qquad \theta^{\mathrm{next}} = \arg\max_{\theta}\;\dfrac{\ell(\theta)}{g(\theta)}$"
EQ["eq_rf"] = r"$\hat{p}_{\mathrm{RF}}(\mathbf{x}) = \dfrac{1}{B}\sum_{b=1}^{B} \hat{p}_{b}(\mathbf{x})$"

EQ["eq_acc"] = r"$\mathrm{Accuracy} = \dfrac{TP+TN}{TP+TN+FP+FN}$"
EQ["eq_pr"] = r"$\mathrm{Precision} = \dfrac{TP}{TP+FP},\qquad \mathrm{Recall} = \dfrac{TP}{TP+FN}$"
EQ["eq_f1"] = r"$F_{1} = 2\cdot\dfrac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}$"
EQ["eq_f1avg"] = r"$F_{1}^{\mathrm{macro}} = \dfrac{1}{2}\sum_{c\in\{0,1\}}F_{1,c},\qquad F_{1}^{\mathrm{weighted}} = \sum_{c}\dfrac{n_c}{n}F_{1,c}$"
EQ["eq_roc1"] = r"$\mathrm{TPR}(t) = \dfrac{TP(t)}{TP(t)+FN(t)},\qquad \mathrm{FPR}(t) = \dfrac{FP(t)}{FP(t)+TN(t)}$"
EQ["eq_roc2"] = r"$\mathrm{AUC} = \int_{0}^{1}\mathrm{TPR}\;d(\mathrm{FPR}) = P\!\left(\hat{p}(\mathbf{x}^{+})>\hat{p}(\mathbf{x}^{-})\right)$"
EQ["eq_tpe1"] = r"$\ell(\theta)=p(\theta\mid y\ \mathrm{termasuk}\ \gamma\%\ \mathrm{terbaik}),\qquad g(\theta)=p(\theta\mid y\ \mathrm{lainnya})$"
EQ["eq_tpe2"] = r"$\theta^{\mathrm{next}} = \arg\max_{\theta}\;\dfrac{\ell(\theta)}{g(\theta)}$"
EQ["eq_gini"] = r"$\mathrm{Gini}(S) = 1-\sum_{c\in\{0,1\}} p_c^{2}$"
EQ["eq_gini2"] = r"$\Delta = \mathrm{Gini}(S)-\dfrac{|S_L|}{|S|}\mathrm{Gini}(S_L)-\dfrac{|S_R|}{|S|}\mathrm{Gini}(S_R)$"
EQ["eq_xgb"] = r"$\mathcal{L} = \sum_{i=1}^{n} w_{y_i}\,\ell\!\left(y_i,\hat{y}_i\right) + \sum_{m=1}^{M}\Omega(f_m)$"
EQ["eq_omega"] = r"$\Omega(f) = \gamma T + \dfrac{1}{2}\lambda\Vert\mathbf{w}\Vert_{2}^{2} + \alpha\Vert\mathbf{w}\Vert_{1}$"
EQ["eq_spw"] = r"$w_{1} = \mathrm{scale\_pos\_weight} = \dfrac{N_{0}}{N_{1}} = \dfrac{2214}{300} = 7.38,\qquad w_{0}=1$"
EQ["eq_grid"] = r"$N_{\mathrm{grid}} = \prod_{j=1}^{d}|V_{j}|,\qquad P(\mathrm{min.\ satu\ trial\ di\ top}\ q) = 1-(1-q)^{n}$"
for name, tex in EQ.items():
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, tex, fontsize=15)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=False,
                facecolor="white")
    plt.close(fig)
print("ok", len(EQ))
