"""Compare tous les runs d'un dossier : tableau, graphes et figure qualitative.

Exemple :  python compare.py --runs-dir runs --data-dir data/isic2018_256
À lancer après test.py (avec --save-pred pour la figure qualitative).
Les fichiers sont écrits dans <runs-dir>/comparison/.
"""
import argparse
import csv
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from mdl_seg.dataset import get_split_indices

# Nom affiché de chaque run. L'ordre fixe aussi la couleur : un run garde toujours la même.
LABELS = {
    "unet": "U-Net",
    "cto": "CTO",
    "cto_no_edge": "CTO sans supervision des contours",
    "cto_background": "CTO, attention sur l'arrière-plan",
    "cto_no_trans": "CTO sans Transformer",
    "cto_light": "CTO, tête légère",
}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_MUTED, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def label_of(name):
    return LABELS.get(name, name)


def color_of(name, extra):
    """Couleur attachée au run (pas à son rang). Les runs inconnus prennent les teintes suivantes."""
    known = list(LABELS)
    index = known.index(name) if name in known else len(known) + extra.index(name)
    return PALETTE[index % len(PALETTE)]


def read_per_image(path):
    """Dice de chaque image, lu dans un test_results.csv."""
    return {row["name"]: float(row["dice"]) for row in csv.DictReader(open(path)) if row["name"] != "MOYENNE"}


def load_runs(runs_dir, fold):
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name, f"fold_{fold}")
        if not os.path.exists(os.path.join(d, "options.json")):
            continue
        run = {"name": name, "dir": d, "summary": None, "log": None, "per_image": {}, "external": {}}
        if os.path.exists(os.path.join(d, "test_summary.json")):
            run["summary"] = json.load(open(os.path.join(d, "test_summary.json")))
        if os.path.exists(os.path.join(d, "test_results.csv")):
            run["per_image"] = read_per_image(os.path.join(d, "test_results.csv"))
        for sub in sorted(os.listdir(d)):
            ext_dir = os.path.join(d, sub)
            if sub.startswith("external_") and os.path.exists(os.path.join(ext_dir, "test_summary.json")):
                run["external"][sub[len("external_"):]] = {
                    "dir": ext_dir,
                    "summary": json.load(open(os.path.join(ext_dir, "test_summary.json"))),
                    "per_image": read_per_image(os.path.join(ext_dir, "test_results.csv")),
                }
        if os.path.exists(os.path.join(d, "log.csv")):
            rows = list(csv.DictReader(open(os.path.join(d, "log.csv"))))
            run["log"] = {key: np.array([float(r[key]) for r in rows]) for key in ("epoch", "loss", "val_dice")}
        runs.append(run)
    return runs


def write_table(runs, out):
    tested = sorted([r for r in runs if r["summary"]], key=lambda r: -r["summary"]["dice"])
    header = ["Modèle", "Dice (%)", "IoU (%)", "Paramètres (M)", "FLOPs (G)"]
    rows = []
    for r in tested:
        s = r["summary"]
        flops = f"{s['flops_G']:.2f}" if s.get("flops_G") is not None else "-"
        rows.append([label_of(r["name"]), f"{100 * s['dice']:.2f}", f"{100 * s['iou']:.2f}", f"{s['params_M']:.2f}", flops])
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    lines = ["| " + " | ".join(header) + " |", "|---|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(out, "results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return tested


def plot_scores(tested, out):
    """Dice et IoU de test, un point par modèle, du meilleur au moins bon.

    Les écarts entre modèles sont de quelques points : des points sur un axe resserré les montrent
    honnêtement, alors que des barres partant de zéro les écraseraient.
    """
    names = [label_of(r["name"]) for r in tested][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(tested) + 1.6), sharey=True)
    for ax, key, title in zip(axes, ("dice", "iou"), ("Dice sur le jeu de test (%)", "IoU sur le jeu de test (%)")):
        values = [100 * r["summary"][key] for r in tested][::-1]
        ax.scatter(values, range(len(values)), s=70, color=PALETTE[0], zorder=3)
        span = max(max(values) - min(values), 1.0)
        ax.set_xlim(min(values) - 0.25 * span, max(values) + 0.35 * span)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        for y, v in enumerate(values):
            ax.annotate(f"{v:.2f}", (v, y), xytext=(9, 0), textcoords="offset points", va="center", color=INK)
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scores.png"), dpi=200)
    plt.close(fig)


def plot_curves(runs, out):
    """Dice de validation au fil des époques, une courbe par run."""
    logged = [r for r in runs if r["log"] is not None and len(r["log"]["epoch"]) > 0]
    if not logged:
        return
    extra = [r["name"] for r in logged if r["name"] not in LABELS]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for r in logged:
        ax.plot(r["log"]["epoch"], 100 * r["log"]["val_dice"], lw=2, color=color_of(r["name"], extra),
                label=f"{label_of(r['name'])} ({100 * r['log']['val_dice'].max():.1f})")
    # Les premières époques partent très bas : on cadre sur la partie utile.
    finals = [100 * np.percentile(r["log"]["val_dice"], 25) for r in logged]
    low = max(0.0, min(finals) - 3)
    ax.set_ylim(low, min(100.0, max(100 * r["log"]["val_dice"].max() for r in logged) + 1))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Époque")
    ax.set_title("Dice de validation pendant l'entraînement (%)")
    ax.legend(title="Run (meilleur Dice)", frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_curves.png"), dpi=200)
    plt.close(fig)


def plot_dice_vs_params(tested, out):
    """Précision contre taille du modèle. Sert de référence pour le pruning."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = [r["summary"]["params_M"] for r in tested]
    ys = [100 * r["summary"]["dice"] for r in tested]
    ax.scatter(xs, ys, s=70, color=PALETTE[0], zorder=3)
    span = max(max(ys) - min(ys), 1.0)
    # Plusieurs runs ont la même taille et des scores proches : on écarte les étiquettes
    # verticalement pour qu'elles ne se chevauchent pas, avec un trait fin vers leur point.
    gap, last = 0.09 * span, -np.inf
    for i in np.argsort(ys):
        y_text = max(ys[i], last + gap)
        last = y_text
        ax.annotate(label_of(tested[i]["name"]), (xs[i], ys[i]), xytext=(xs[i] + 0.03 * max(xs), y_text),
                    va="center", color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.6, shrinkA=0, shrinkB=5))
    ax.set_xlim(0, max(xs) * 1.6)
    ax.set_ylim(min(ys) - 0.3 * span, max(last, max(ys)) + 0.2 * span)
    ax.set_xlabel("Paramètres (millions)")
    ax.set_title("Dice de test (%) selon la taille du modèle")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "dice_vs_params.png"), dpi=200)
    plt.close(fig)


def plot_qualitative(entries, data_dir, indices, n_images, path, where):
    """Image, vérité terrain et prédiction de chaque modèle, comme les figures du papier.

    entries : liste de dictionnaires {name, dir, per_image, dice}, un par modèle.
    On choisit les images où les modèles diffèrent le plus (écart de Dice le plus grand).
    """
    shown = [e for e in entries if os.path.isdir(os.path.join(e["dir"], "predictions")) and e["per_image"]]
    if not shown:
        print(f"Figure qualitative ({where}) ignorée : lance d'abord test.py avec --save-pred.")
        return
    shown.sort(key=lambda e: -e["dice"])
    names = np.load(os.path.join(data_dir, "filename.npy"))
    imgs = np.load(os.path.join(data_dir, "img.npy"), mmap_mode="r")
    masks = np.load(os.path.join(data_dir, "mask.npy"), mmap_mode="r")
    common = [i for i in indices(len(names)) if all(str(names[i]) in e["per_image"] for e in shown)]
    if len(shown) > 1:
        spread = [np.ptp([e["per_image"][str(names[i])] for e in shown]) for i in common]
        chosen = [common[j] for j in np.argsort(spread)[::-1][:n_images]]
        how = "où les modèles diffèrent le plus"
    else:  # un seul modèle : ses images les moins bien segmentées
        scores = [shown[0]["per_image"][str(names[i])] for i in common]
        chosen = [common[j] for j in np.argsort(scores)[:n_images]]
        how = "les moins bien segmentées"

    cols = 2 + len(shown)
    fig, axes = plt.subplots(len(chosen), cols, figsize=(2.3 * cols, 2.45 * len(chosen)), squeeze=False)
    for row, i in enumerate(chosen):
        name = str(names[i])
        gt = np.array(masks[i])
        panels = [(np.array(imgs[i]), "Image"), (gt, "Vérité terrain")]
        for e in shown:
            pred = cv2.imread(os.path.join(e["dir"], "predictions", f"{name}_pred.png"), cv2.IMREAD_GRAYSCALE)
            panels.append(((pred > 127).astype(np.uint8), label_of(e["name"])))
        for col, (panel, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1 if panel.ndim == 2 else None)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=9, loc="center", wrap=True)
            if col >= 2:
                # contour de la vérité terrain par-dessus la prédiction + Dice de l'image
                if gt.min() != gt.max():
                    ax.contour(gt, levels=[0.5], colors=[PALETTE[1]], linewidths=1.2)
                ax.set_xlabel(f"Dice {100 * shown[col - 2]['per_image'][name]:.1f}", fontsize=9, color=INK)
    fig.suptitle(f"Prédictions sur les images de {where} {how} "
                 "(trait orange : contour de la vérité terrain)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_generalization(runs, ext, out):
    """Tableau et graphe : score sur le jeu de test d'origine contre score sur le jeu externe.

    Le modèle n'a jamais vu le jeu externe. Un écart faible indique une bonne généralisation.
    """
    both = sorted([r for r in runs if r["summary"] and ext in r["external"]], key=lambda r: -r["summary"]["dice"])
    if not both:
        return
    header = ["Modèle", "Dice ISIC 2018 (%)", f"Dice {ext.upper()} (%)", "Écart (points)",
              "IoU ISIC 2018 (%)", f"IoU {ext.upper()} (%)"]
    rows = []
    for r in both:
        a, b = r["summary"], r["external"][ext]["summary"]
        rows.append([label_of(r["name"]), f"{100 * a['dice']:.2f}", f"{100 * b['dice']:.2f}",
                     f"{100 * (b['dice'] - a['dice']):+.2f}", f"{100 * a['iou']:.2f}", f"{100 * b['iou']:.2f}"])
    with open(os.path.join(out, f"generalization_{ext}.csv"), "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    lines = ["| " + " | ".join(header) + " |", "|---|---|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(out, f"generalization_{ext}.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # Un trait par modèle, reliant son Dice sur ISIC (bleu) à son Dice sur le jeu externe (orange).
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(both) + 1.9))
    ys = range(len(both))[::-1]
    values = []
    for y, r in zip(ys, both):
        a, b = 100 * r["summary"]["dice"], 100 * r["external"][ext]["summary"]["dice"]
        values += [a, b]
        ax.plot([a, b], [y, y], color=INK_MUTED, lw=1.2, zorder=2)
        ax.scatter([a], [y], s=70, color=PALETTE[0], zorder=3, label="ISIC 2018 (jeu de test)" if y == len(both) - 1 else None)
        ax.scatter([b], [y], s=70, color=PALETTE[1], zorder=3, label=f"{ext.upper()} (jamais vu)" if y == len(both) - 1 else None)
        left, right = (a, b) if a <= b else (b, a)
        ax.annotate(f"{left:.1f}", (left, y), xytext=(-9, 0), textcoords="offset points", ha="right", va="center", color=INK)
        ax.annotate(f"{right:.1f}", (right, y), xytext=(9, 0), textcoords="offset points", ha="left", va="center", color=INK)
    span = max(max(values) - min(values), 1.0)
    ax.set_xlim(min(values) - 0.3 * span, max(values) + 0.3 * span)
    ax.set_yticks(list(ys), [label_of(r["name"]) for r in both])
    ax.set_ylim(-0.7, len(both) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Généralisation : Dice (%) sur ISIC 2018 et sur {ext.upper()}, sans réentraînement")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"generalization_{ext}.png"), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--data-dir", default="data/isic2018_256")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-images", type=int, default=6, help="nombre d'images de la figure qualitative")
    p.add_argument("--external-dir", default=None, help="données du jeu externe, pour sa figure qualitative")
    p.add_argument("--external-name", default="ph2", help="nom du jeu externe (comme dans test.py)")
    args = p.parse_args()

    out = os.path.join(args.runs_dir, "comparison")
    os.makedirs(out, exist_ok=True)
    runs = [r for r in load_runs(args.runs_dir, args.fold)]
    assert runs, f"aucun run trouvé dans {args.runs_dir}"

    tested = write_table(runs, out)
    plot_curves(runs, out)
    if tested:
        plot_scores(tested, out)
        plot_dice_vs_params(tested, out)
    else:
        print("Aucun run testé : lance test.py pour obtenir le tableau et les graphes de scores.")
    entries = [{"name": r["name"], "dir": r["dir"], "per_image": r["per_image"],
                "dice": (r["summary"] or {}).get("dice", 0)} for r in runs]
    plot_qualitative(entries, args.data_dir, lambda n: get_split_indices(n, args.fold, "test"),
                     args.n_images, os.path.join(out, "qualitative.png"), "test")

    # Jeux externes (ex. PH2) : généralisation sans réentraînement
    for ext in sorted({name for r in runs for name in r["external"]}):
        write_generalization(runs, ext, out)
        if args.external_dir and ext == args.external_name:
            entries = [{"name": r["name"], "dir": r["external"][ext]["dir"], "per_image": r["external"][ext]["per_image"],
                        "dice": r["external"][ext]["summary"]["dice"]} for r in runs if ext in r["external"]]
            plot_qualitative(entries, args.external_dir, lambda n: np.arange(n), args.n_images,
                             os.path.join(out, f"qualitative_{ext}.png"), ext.upper())
    print(f"=> fichiers écrits dans {out}")


if __name__ == "__main__":
    main()
