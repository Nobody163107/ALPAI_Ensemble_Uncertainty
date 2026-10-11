"""E4 - Images: 10 CNNs trained on MNIST, shown things they have never seen.

* in-distribution quality: accuracy, NLL, Brier, calibration (ECE)
* out-of-distribution detection: MNIST test (in) vs Fashion-MNIST test (out), AUROC
* dataset shift: rotate the digits by 0..180 degrees
* how many members do we need? M = 1..10
* MC Dropout (implicit ensemble, same test-time cost) as a comparison

The first run trains the 11 networks (about 20-30 min on a laptop CPU) and stores them
in checkpoints/; later runs just load them.

    python experiments/e4_mnist_ood.py            # train if needed, evaluate, plot
    python experiments/e4_mnist_ood.py --retrain  # force retraining
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import CKPT_DIR, FIG_DIR, save_json

from ensemble_uq import DeepEnsemble, MCEnsemble, decompose_classification
from ensemble_uq.data import ASSETS_DIR, FASHION_CLASSES, load_fashion_mnist, load_mnist, normalize, rotate
from ensemble_uq.metrics import accuracy, brier, expected_calibration_error, nll, ood_auroc, reliability_curve
from ensemble_uq.models import SmallCNN
from ensemble_uq.plotting import (ALEATORIC, EPISTEMIC, INK, INK2, METHOD_COLORS, MUTED, SEEN, UNSEEN,
                                  set_style)
from ensemble_uq.training import fit_classifier

N_MEMBERS = 10
EPOCHS = 10
MC_P = 0.25
MC_SAMPLES = 10
ANGLES = list(range(0, 181, 15))


def member_path(m: int):
    return CKPT_DIR / f"mnist_cnn_member{m}.pt"


def train_or_load(Xtr, ytr, retrain: bool = False, log=print):
    members = []
    for m in range(N_MEMBERS):
        model = SmallCNN()
        if member_path(m).exists() and not retrain:
            model.load_state_dict(torch.load(member_path(m), map_location="cpu"))
        else:
            t = time.time()
            torch.manual_seed(m)                     # different random initialisation per member
            model = SmallCNN()
            fit_classifier(model, normalize(Xtr), ytr, epochs=EPOCHS, lr=1e-3, weight_decay=0.0,
                           batch_size=128, seed=m)   # different mini-batch order per member
            torch.save(model.state_dict(), member_path(m))
            log(f"member {m + 1}/{N_MEMBERS} trained in {time.time() - t:.0f}s")
        members.append(model.eval())

    mc_path = CKPT_DIR / "mnist_cnn_mcdropout.pt"
    mc_model = SmallCNN(dropout=MC_P)
    if mc_path.exists() and not retrain:
        mc_model.load_state_dict(torch.load(mc_path, map_location="cpu"))
    else:
        t = time.time()
        torch.manual_seed(100)
        mc_model = SmallCNN(dropout=MC_P)
        fit_classifier(mc_model, normalize(Xtr), ytr, epochs=EPOCHS, lr=1e-3, weight_decay=0.0,
                       batch_size=128, seed=100)
        torch.save(mc_model.state_dict(), mc_path)
        log(f"MC-dropout network trained in {time.time() - t:.0f}s")
    return DeepEnsemble(members=members), MCEnsemble(model=mc_model.eval(), n_samples=MC_SAMPLES)


def in_dist_metrics(probs, y):
    return {"accuracy": accuracy(probs, y), "nll": nll(probs, y), "brier": brier(probs, y),
            "ece": expected_calibration_error(probs, y)}


def run(retrain: bool = False):
    set_style()
    t0 = time.time()
    Xtr, ytr = load_mnist(train=True)
    Xte, yte = load_mnist(train=False)
    Xood, yood = load_fashion_mnist(train=False)
    y_np = yte.numpy()

    de, mc = train_or_load(Xtr, ytr, retrain)
    P_in = de.member_probs(normalize(Xte))     # (10, 10000, 10)
    P_out = de.member_probs(normalize(Xood))
    MC_in = mc.member_probs(normalize(Xte))
    MC_out = mc.member_probs(normalize(Xood))
    print(f"predictions done ({time.time() - t0:.0f}s)")

    # ---------------- in-distribution quality ----------------------------------------
    table = {
        "Single network": in_dist_metrics(P_in[0], y_np),
        "Deep Ensemble (M=5)": in_dist_metrics(P_in[:5].mean(0), y_np),
        "Deep Ensemble (M=10)": in_dist_metrics(P_in.mean(0), y_np),
        f"MC Dropout (T={MC_SAMPLES})": in_dist_metrics(MC_in.mean(0), y_np),
    }

    # ---------------- OOD detection ----------------------------------------------------
    d_in, d_out = decompose_classification(P_in), decompose_classification(P_out)
    s_in, s_out = decompose_classification(P_in[:1]), decompose_classification(P_out[:1])
    m_in, m_out = decompose_classification(MC_in), decompose_classification(MC_out)
    auroc = {
        "Single network": {"total": ood_auroc(s_in.total, s_out.total),
                           "max-prob": ood_auroc(-s_in.confidence, -s_out.confidence)},
        "Deep Ensemble (M=10)": {k: ood_auroc(getattr(d_in, k), getattr(d_out, k))
                                 for k in ("total", "aleatoric", "epistemic")},
        f"MC Dropout (T={MC_SAMPLES})": {k: ood_auroc(getattr(m_in, k), getattr(m_out, k))
                                        for k in ("total", "aleatoric", "epistemic")},
    }
    single_conf_out = s_out.confidence
    overconfident_share = float((single_conf_out > 0.99).mean())

    # ---------------- ensemble size ------------------------------------------------------
    sizes = list(range(1, N_MEMBERS + 1))
    by_size = {"M": sizes, "accuracy": [], "nll": [], "ece": [], "auroc_total": [], "auroc_epistemic": []}
    for k in sizes:
        a, b = decompose_classification(P_in[:k]), decompose_classification(P_out[:k])
        p = P_in[:k].mean(0)
        by_size["accuracy"].append(accuracy(p, y_np))
        by_size["nll"].append(nll(p, y_np))
        by_size["ece"].append(expected_calibration_error(p, y_np))
        by_size["auroc_total"].append(ood_auroc(a.total, b.total))
        by_size["auroc_epistemic"].append(ood_auroc(a.epistemic, b.epistemic) if k > 1 else None)

    # ---------------- rotation ------------------------------------------------------------
    Xr, yr = Xte[:2000], y_np[:2000]
    rot = {"angles": ANGLES, "acc_single": [], "acc_ensemble": [], "acc_mc": [], "total": [], "aleatoric": [],
           "epistemic": [], "single_entropy": []}
    for ang in ANGLES:
        P = de.member_probs(normalize(rotate(Xr, ang)))
        Pm = mc.member_probs(normalize(rotate(Xr, ang)))
        d = decompose_classification(P)
        rot["acc_single"].append(accuracy(P[0], yr))
        rot["acc_ensemble"].append(accuracy(P.mean(0), yr))
        rot["acc_mc"].append(accuracy(Pm.mean(0), yr))
        rot["single_entropy"].append(float(decompose_classification(P[:1]).total.mean()))
        for k in ("total", "aleatoric", "epistemic"):
            rot[k].append(float(getattr(d, k).mean()))
    print(f"rotation done ({time.time() - t0:.0f}s)")

    # ---------------- figures --------------------------------------------------------------
    hook = fig_overconfident(Xood, yood.numpy(), P_out, s_out, d_out)
    fig_ood_histograms(s_in, s_out, d_in, d_out, auroc)
    fig_rotation(Xte, rot)
    fig_ensemble_size(by_size)
    gallery = fig_gallery(Xte, y_np, d_in)
    fig_reliability(P_in, y_np)
    save_demo_samples(Xte, y_np, Xood, yood.numpy())

    bins = np.linspace(0, np.log2(10), 41)
    hist = {"bins": bins,
            "mnist_epistemic": np.histogram(d_in.epistemic, bins=bins, density=True)[0],
            "fashion_epistemic": np.histogram(d_out.epistemic, bins=bins, density=True)[0],
            "mnist_epistemic_quantiles": np.quantile(d_in.epistemic, [0.5, 0.9, 0.95, 0.99]),
            "fashion_epistemic_quantiles": np.quantile(d_out.epistemic, [0.05, 0.1, 0.5])}
    out = {"n_members": N_MEMBERS, "epochs": EPOCHS, "mc_dropout_p": MC_P, "mc_samples": MC_SAMPLES,
           "in_distribution": table, "ood_auroc": auroc, "ensemble_size": by_size, "rotation": rot,
           "single_overconfident_on_fashion_share": overconfident_share, "hook_example": hook,
           "gallery": gallery, "epistemic_histogram": hist,
           "mean_uncertainty": {"mnist": {k: float(getattr(d_in, k).mean()) for k in ("total", "aleatoric", "epistemic")},
                                "fashion": {k: float(getattr(d_out, k).mean()) for k in ("total", "aleatoric", "epistemic")}}}
    save_json(out, "e4_mnist.json")
    print(table)
    print(auroc)
    print(f"E4 done in {time.time() - t0:.0f}s")
    return out


# =============================================================================== figures
def fig_overconfident(Xood, yood, P_out, s_out, d_out):
    """Pick a Fashion-MNIST image the single network is >99 % sure is a digit, where members disagree."""
    sure = np.where(s_out.confidence > 0.99)[0]
    idx = int(sure[np.argmax(d_out.epistemic[sure])]) if len(sure) else int(np.argmax(d_out.epistemic))
    probs = P_out[:, idx, :]
    fig = plt.figure(figsize=(7.6, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.45, 2.0], wspace=0.32)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(Xood[idx, 0], cmap="gray_r")
    ax0.set_xticks([]), ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_visible(True), spine.set_color("#c3c2b7")
    ax0.set_title("input", fontsize=10)
    ax1 = fig.add_subplot(gs[1])
    ax1.bar(range(10), probs[0], color=MUTED, width=0.7)
    ax1.grid(axis="x", visible=False)
    ax1.set_xticks(range(10))
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"one network: '{probs[0].argmax()}' at {probs[0].max():.1%}", fontsize=10)
    ax1.set_xlabel("digit class")
    ax2 = fig.add_subplot(gs[2])
    im = ax2.imshow(probs, cmap="Purples", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(len(probs)))
    ax2.set_yticklabels([f"h{m + 1}" for m in range(len(probs))], fontsize=7)
    ax2.grid(False)
    ax2.set_xlabel("digit class")
    ax2.set_title(f"10 members disagree: EU = {d_out.epistemic[idx]:.2f} bits", fontsize=10)
    cb = plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    fig.savefig(FIG_DIR / "e4_overconfident.png")
    plt.close(fig)
    # standalone pieces for slides
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.imshow(Xood[idx, 0], cmap="gray_r")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True), s.set_color("#c3c2b7")
    fig.savefig(FIG_DIR / "e4_hook_image.png")
    plt.close(fig)
    return {"index": idx, "true_fashion_class": FASHION_CLASSES[int(yood[idx])], "member_probs": probs,
            "single_pred": int(probs[0].argmax()), "single_conf": float(probs[0].max()),
            "ensemble_mean": probs.mean(0), "ensemble_pred": int(probs.mean(0).argmax()),
            "ensemble_conf": float(probs.mean(0).max()), "member_preds": probs.argmax(1),
            "total": float(d_out.total[idx]), "aleatoric": float(d_out.aleatoric[idx]),
            "epistemic": float(d_out.epistemic[idx])}


def fig_ood_histograms(s_in, s_out, d_in, d_out, auroc):
    bins = np.linspace(0, np.log2(10), 41)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    panels = [(axes[0], s_in.total, s_out.total, "Single network: entropy",
               auroc["Single network"]["total"]),
              (axes[1], d_in.epistemic, d_out.epistemic, "Deep Ensemble: epistemic EU(x)",
               auroc["Deep Ensemble (M=10)"]["epistemic"])]
    for ax, a, b, title, au in panels:
        ax.hist(a, bins=bins, density=True, color=SEEN, alpha=0.9, label="MNIST test (seen kind)")
        ax.hist(b, bins=bins, density=True, color=UNSEEN, alpha=0.55, label="Fashion-MNIST (never seen)")
        ax.set_yscale("log")
        ax.set_title(f"{title}   (AUROC {au:.3f})", fontsize=11)
        ax.set_xlabel("uncertainty (bits)")
    axes[0].set_ylabel("density (log scale)")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e4_ood_histograms.png")
    plt.close(fig)


def fig_rotation(Xte, rot):
    fig = plt.figure(figsize=(10.5, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.55, 3], hspace=0.15, wspace=0.28)
    strip = fig.add_subplot(gs[0, :])
    digit = Xte[0:1]
    imgs = [rotate(digit, a)[0, 0].numpy() for a in rot["angles"]]
    strip.imshow(np.concatenate([np.pad(im, 2, constant_values=0) for im in imgs], axis=1), cmap="gray_r")
    strip.axis("off")
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(rot["angles"], rot["acc_ensemble"], marker="o", ms=5, color=METHOD_COLORS["Deep Ensemble"],
             label="Deep Ensemble (M=10)")
    ax1.plot(rot["angles"], rot["acc_single"], marker="o", ms=5, color=MUTED, label="single network")
    ax1.set_xlabel("rotation angle (degrees)")
    ax1.set_ylabel("accuracy")
    ax1.set_xticks(rot["angles"][::2])
    ax1.legend(fontsize=9)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(rot["angles"], rot["aleatoric"], marker="o", ms=5, color=ALEATORIC, label="aleatoric AU")
    ax2.plot(rot["angles"], rot["epistemic"], marker="o", ms=5, color=EPISTEMIC, label="epistemic EU")
    ax2.set_xlabel("rotation angle (degrees)")
    ax2.set_ylabel("mean uncertainty (bits)")
    ax2.set_xticks(rot["angles"][::2])
    ax2.legend(fontsize=9)
    fig.savefig(FIG_DIR / "e4_rotation.png")
    plt.close(fig)


def fig_ensemble_size(by_size):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    de_colour = METHOD_COLORS["Deep Ensemble"]
    axes[0].plot(by_size["M"], by_size["nll"], marker="o", ms=6, color=de_colour)
    axes[0].set_title("Test NLL on MNIST (lower is better)", fontsize=11)
    axes[0].set_xlabel("ensemble size M")
    axes[1].plot(by_size["M"][1:], by_size["auroc_epistemic"][1:], marker="o", ms=6, color=de_colour)
    axes[1].set_title("OOD detection AUROC with EU (higher is better)", fontsize=11)
    axes[1].set_xlabel("ensemble size M")
    for ax in axes:
        ax.set_xticks(by_size["M"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e4_ensemble_size.png")
    plt.close(fig)


def fig_gallery(Xte, y_np, d_in, k: int = 7):
    top_au = np.argsort(-(d_in.aleatoric - d_in.epistemic))[:k]
    top_eu = np.argsort(-(d_in.epistemic - d_in.aleatoric))[:k]
    fig, axes = plt.subplots(2, k, figsize=(7.6, 2.45))
    for row, (idxs, colour, label) in enumerate(((top_au, ALEATORIC, "aleatoric"),
                                                 (top_eu, EPISTEMIC, "epistemic"))):
        for ax, i in zip(axes[row], idxs):
            ax.imshow(Xte[i, 0], cmap="gray_r")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True), s.set_color(colour), s.set_linewidth(2)
            ax.set_title(f"true {y_np[i]}, pred {d_in.prediction[i]}", fontsize=9, color=INK2)
        axes[row, 0].set_ylabel(label, fontsize=10, color=colour, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e4_gallery.png")
    plt.close(fig)
    return {"top_aleatoric": top_au, "top_epistemic": top_eu}


def fig_reliability(P_in, y_np):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, probs, title, colour in ((axes[0], P_in[0], "Single network", MUTED),
                                     (axes[1], P_in.mean(0), "Deep Ensemble (M=10)", METHOD_COLORS["Deep Ensemble"])):
        conf, acc, count = reliability_curve(probs, y_np, 15)
        keep = count > 0
        ax.plot([0, 1], [0, 1], color=INK, ls="--", lw=1)
        ax.plot(conf[keep], acc[keep], marker="o", ms=5, color=colour)
        ax.set_title(f"{title}: ECE {expected_calibration_error(probs, y_np):.4f}", fontsize=11)
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e4_reliability.png")
    plt.close(fig)


def save_demo_samples(Xte, y_np, Xood, yood, n: int = 300):
    """A small offline subset for the live app (works without internet)."""
    rng = np.random.default_rng(0)
    i_in = np.sort(rng.choice(len(Xte), n, replace=False))
    i_out = np.sort(rng.choice(len(Xood), n, replace=False))
    np.savez_compressed(ASSETS_DIR / "demo_samples.npz",
                        mnist_x=(Xte[i_in, 0].numpy() * 255).astype(np.uint8), mnist_y=y_np[i_in],
                        fashion_x=(Xood[i_out, 0].numpy() * 255).astype(np.uint8), fashion_y=yood[i_out],
                        fashion_classes=np.array(FASHION_CLASSES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    run(retrain=parser.parse_args().retrain)
