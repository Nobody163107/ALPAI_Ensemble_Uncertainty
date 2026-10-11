"""Live demo - uncertainty from ensembles (aleatoric vs epistemic).

    streamlit run app/streamlit_app.py

Works offline: the image tab uses the trained CNNs in checkpoints/ and a small bundle of
test images in assets/demo_samples.npz. Each tab is a Streamlit *fragment*, so moving a
slider only re-runs that tab.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from ensemble_uq import (DeepEnsemble, GaussianEnsemble, LaplaceRandomForest, MCEnsemble,  # noqa: E402
                         ParallelMLPEnsemble, decompose_classification, member_table)
from ensemble_uq.data import (REGRESSION_SUPPORT, grid_2d, load_demo_samples, make_regression_1d,  # noqa: E402
                              make_toy_2d, normalize, regression_noise_std, regression_truth, rotate)
from ensemble_uq.models import MLP, SmallCNN  # noqa: E402
from ensemble_uq.plotting import (INK, METHOD_COLORS, MUTED, YELLOW, member_boundaries, regression_panels,  # noqa: E402
                                  set_style, uncertainty_map)

torch.set_num_threads(1)  # tiny models: one thread is fastest and keeps the laptop responsive
set_style(10)
LIM = 6.0
RES = 140

st.set_page_config(page_title="Ensemble uncertainty demo", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 1rem;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      .au {color: #c24e1c; font-weight: 700;}
      .eu {color: #2a78d6; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Where is the model unsure - and why?")
st.markdown(
    "Uncertainty from a committee of models: <span class='au'>aleatoric</span> = the members agree the "
    "answer is a coin flip; <span class='eu'>epistemic</span> = the members disagree with each other.",
    unsafe_allow_html=True,
)


def star(ax, x, y):
    ax.scatter([x], [y], s=170, marker="*", color=YELLOW, edgecolors=INK, linewidths=0.8, zorder=6)


# ===================================================================== 2D classification
@st.cache_resource(show_spinner="Training the ensemble...")
def fit_2d(kind, n, noise, hole, method, n_members, seed):
    X, y = make_toy_2d(kind, n=n, noise=noise, seed=seed, hole=hole)
    if method == "Deep Ensemble":
        model = ParallelMLPEnsemble(2, 2, n_members=n_members, seed=seed).fit(X, y)
    elif method == "Bagging (bootstrap)":
        model = ParallelMLPEnsemble(2, 2, n_members=n_members, bootstrap=True, seed=seed).fit(X, y)
    elif method == "MC DropConnect":
        model = MCEnsemble(lambda: MLP(2, 2, hidden=(64, 64), dropconnect=0.2), n_samples=max(n_members, 2),
                           seed=seed, batch_size=100_000).fit(X, y, epochs=500, lr=1e-2, weight_decay=1e-4)
    else:
        model = LaplaceRandomForest(n_estimators=n_members, min_samples_leaf=5, seed=seed).fit(X, y)
    _, _, grid = grid_2d(LIM, RES)
    return X, y, model, model.member_probs(grid)


@st.fragment
def view_2d():
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.2, 1.1])
    kind = c1.selectbox("Data set", ["blobs", "moons", "circles", "xor"],
                        help="blobs = two overlapping Gaussian classes (lots of aleatoric uncertainty)")
    n = c1.slider("Training points", 50, 800, 400, step=50)
    noise = c2.slider("Class spread / noise", 0.4, 1.6, 1.0, step=0.1)
    use_hole = c2.checkbox("Cut a hole in the data", value=False,
                           help="Removes all training points within a disc - watch EU appear there.")
    hole_x = c2.slider("hole centre x", -4.0, 4.0, 0.0, step=0.5, disabled=not use_hole)
    method = c3.radio("How the ensemble is built",
                      ["Deep Ensemble", "Bagging (bootstrap)", "MC DropConnect", "Random Forest"])
    max_m = 100 if method == "Random Forest" else 20
    n_members = c3.slider("Members M (trees for the forest, passes for DropConnect)", 1, max_m,
                          min(10, max_m), help="M = 1 is a single model: epistemic uncertainty is always 0.")
    seed = int(c4.number_input("Random seed", 0, 99, 0))
    qx = c4.slider("query point x", -LIM, LIM, 0.0, step=0.25)
    qy = c4.slider("query point y", -LIM, LIM, 5.0, step=0.25)

    hole = (hole_x, 0.0, 1.5) if use_hole else None
    X, y, model, member_p = fit_2d(kind, n, noise, hole, method, n_members, seed)
    dec = decompose_classification(member_p)
    xx, yy, _ = grid_2d(LIM, RES)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    for ax, k in zip(axes[:3], ("total", "aleatoric", "epistemic")):
        uncertainty_map(ax, xx, yy, getattr(dec, k), k, X, y)
        star(ax, qx, qy)
    if method == "Random Forest":
        uncertainty_map(axes[3], xx, yy, dec.mean_probs[:, 1], "epistemic", X, y, title="p(black | x)")
    else:
        member_boundaries(axes[3], xx, yy, member_p, X, y)
        axes[3].set_title(f"{len(member_p)} members' decision boundaries")
    star(axes[3], qx, qy)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    q_probs = model.member_probs(np.array([[qx, qy]], dtype=np.float32))[:, 0, :]
    table, summary = member_table(q_probs, class_names=["p(red)", "p(black)"])
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(f"**The ensemble table at the query point ★ ({qx:.2f}, {qy:.2f})** - lecture slide 33")
        st.dataframe(table.style.format("{:.3f}"), height=min(38 * (len(table) + 1), 420), width="stretch")
    with right:
        u, au, eu = summary.values()
        st.metric("Total U(x) = entropy of the average", f"{u:.3f} bits")
        st.metric("Aleatoric AU(x) = average of the entropies", f"{au:.3f} bits")
        st.metric("Epistemic EU(x) = U - AU", f"{eu:.3f} bits")
        st.caption("Try: move ★ into the overlap (AU up), then far up or down (EU up). "
                   "Set M = 1 and EU disappears - a single model cannot know what it doesn't know.")


# ===================================================================== regression
@st.cache_resource(show_spinner="Training the regression ensemble...")
def fit_reg(n, gap, n_members, seed):
    support = REGRESSION_SUPPORT if gap else ((-4.0, 4.0),)
    x, y = make_regression_1d(n, seed=seed, support=support)
    model = GaussianEnsemble(n_members=n_members, seed=seed).fit(x, y)
    return x, y, model, support


@st.fragment
def view_regression():
    c1, c2, c3 = st.columns(3)
    n_reg = c1.slider("Training points N", 20, 800, 200, step=20)
    gap = c2.checkbox("Leave a gap in the data (-2 < x < 1.5)", value=True)
    m_reg = c3.slider("Members M ", 1, 10, 5)
    seed_reg = int(c3.number_input("Random seed ", 0, 99, 0))
    x, y, reg, support = fit_reg(n_reg, gap, m_reg, seed_reg)
    x_grid = np.linspace(-7, 7, 500).astype(np.float32)
    means, _ = reg.member_moments(x_grid)
    d = reg.predict(x_grid)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    regression_panels(ax1, ax2, x_grid, d, x, y, truth=regression_truth(x_grid),
                      noise_std=regression_noise_std(x_grid), member_means=means, support=support)
    ax1.set_ylim(-4.2, 4.2)
    ax1.legend(loc="lower right", fontsize=8, ncol=3)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper center", fontsize=8, ncol=3)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    in_data = np.zeros_like(x_grid, dtype=bool)
    for a, b in support:
        in_data |= (x_grid > a) & (x_grid < b)
    k1, k2, k3 = st.columns(3)
    k1.metric("Aleatoric σ where we have data", f"{d.aleatoric_std[in_data].mean():.3f}")
    k2.metric("Epistemic σ where we have data", f"{d.epistemic_std[in_data].mean():.3f}")
    k3.metric("Epistemic σ outside the data (|x| > 4.5)", f"{d.epistemic_std[np.abs(x_grid) > 4.5].mean():.3f}")
    st.caption("Raise N or untick the gap: the blue (epistemic) part shrinks where data arrives; "
               "the orange (aleatoric) part stays at the true noise level.")


# ===================================================================== images
@st.cache_resource(show_spinner="Loading the 10 CNNs...")
def load_cnns():
    members = []
    for m in range(10):
        net = SmallCNN()
        net.load_state_dict(torch.load(ROOT / "checkpoints" / f"mnist_cnn_member{m}.pt", map_location="cpu"))
        members.append(net.eval())
    mc = SmallCNN(dropout=0.25)
    mc.load_state_dict(torch.load(ROOT / "checkpoints" / "mnist_cnn_mcdropout.pt", map_location="cpu"))
    return DeepEnsemble(members=members), MCEnsemble(model=mc.eval(), n_samples=10)


@st.cache_data(show_spinner="Scoring the reference images...")
def reference_eu():
    """Epistemic uncertainty of the 10-CNN ensemble on the bundled MNIST test digits."""
    de, _ = load_cnns()
    X = torch.as_tensor(load_demo_samples()["mnist_x"], dtype=torch.float32).div(255).unsqueeze(1)
    return decompose_classification(de.member_probs(normalize(X))).epistemic


def to_tensor(img_uint8):
    return torch.as_tensor(img_uint8, dtype=torch.float32).div(255).reshape(1, 1, 28, 28)


@st.fragment
def view_images():
    if not (ROOT / "checkpoints" / "mnist_cnn_member0.pt").exists():
        st.warning("No trained CNNs found. Run  `python experiments/e4_mnist_ood.py`  once (about 25 min).")
        return
    de, mc = load_cnns()
    samples = load_demo_samples()
    fashion_names = [str(c) for c in samples["fashion_classes"]]
    n_img = len(samples["mnist_y"])
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    source = c1.radio("Show the ensemble...", ["an MNIST digit (seen kind)", "a Fashion-MNIST item (never seen)",
                                               "two digits blended"], index=0)
    idx = c1.slider("image #", 0, n_img - 1, 0)
    angle = c2.slider("rotate (degrees)", 0, 180, 0, step=5)
    two = source.startswith("two")
    blend_idx = c2.slider("second digit # (for blending)", 0, n_img - 1, 7, disabled=not two)
    alpha = c2.slider("blend weight", 0.0, 1.0, 0.5, step=0.05, disabled=not two)
    approach = c3.radio("Ensemble", ["Deep Ensemble (10 CNNs)", "MC Dropout (1 CNN, 10 passes)"])
    m_img = c3.slider("members used", 1, 10, 10, disabled=approach.startswith("MC"))

    if source.startswith("an MNIST"):
        img = to_tensor(samples["mnist_x"][idx])
        caption = f"MNIST test digit, label {int(samples['mnist_y'][idx])}"
    elif source.startswith("a Fashion"):
        img = to_tensor(samples["fashion_x"][idx])
        caption = f"Fashion-MNIST: {fashion_names[int(samples['fashion_y'][idx])]}"
    else:
        img = alpha * to_tensor(samples["mnist_x"][idx]) + (1 - alpha) * to_tensor(samples["mnist_x"][blend_idx])
        caption = f"blend of a {int(samples['mnist_y'][idx])} and a {int(samples['mnist_y'][blend_idx])}"
    if angle:
        img = rotate(img, angle)
        caption += f", rotated {angle}°"

    ens = de.subset(m_img) if approach.startswith("Deep") else mc
    P = ens.member_probs(normalize(img))[:, 0, :]
    dec = decompose_classification(P)

    col_img, col_bar, col_members = st.columns([0.8, 1.3, 1.6])
    with col_img:
        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        ax.imshow(img[0, 0].numpy(), cmap="gray_r", vmin=0, vmax=1)
        ax.set_axis_off()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption(caption)
    with col_bar:
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        ax.bar(range(10), dec.mean_probs[0], color=METHOD_COLORS["Deep Ensemble"], width=0.7, label="ensemble average")
        ax.bar(range(10), P[0], color="none", edgecolor=MUTED, width=0.7, lw=1.5, ls="--", label="member 1 alone")
        ax.set_xticks(range(10))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("digit class")
        ax.set_title(f"prediction: {dec.prediction[0]}  ({dec.confidence[0]:.0%})")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with col_members:
        fig, ax = plt.subplots(figsize=(5.4, 2.9))
        ax.imshow(P, cmap="Purples", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(10))
        ax.set_yticks(range(len(P)))
        ax.set_yticklabels([f"h{m + 1}" for m in range(len(P))], fontsize=8)
        ax.grid(False)
        ax.set_xlabel("digit class")
        ax.set_title("each member's p(y | x, h_m)")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total U(x)", f"{dec.total[0]:.2f} bits")
    k2.metric("Aleatoric AU(x)", f"{dec.aleatoric[0]:.2f} bits")
    k3.metric("Epistemic EU(x)", f"{dec.epistemic[0]:.2f} bits")
    pct = float((reference_eu() < dec.epistemic[0]).mean())
    k4.metric("EU higher than", f"{pct:.0%} of MNIST digits",
              help="Compared with the 300 bundled MNIST test digits, scored by the 10 CNNs.")
    st.caption("Try: rotate a digit past ~45° and watch EU climb; switch to a Fashion item; set members used = 1.")


tab_2d, tab_reg, tab_img = st.tabs(["2D classification", "Regression", "Images: MNIST vs the unknown"])
with tab_2d:
    view_2d()
with tab_reg:
    view_regression()
with tab_img:
    view_images()

with st.expander("The maths on one screen"):
    st.latex(r"p(y\mid x) \approx \frac{1}{M}\sum_{m=1}^{M} p(y\mid x, h_m)")
    st.latex(r"\underbrace{H\Big[\tfrac{1}{M}\textstyle\sum_m p(y\mid x,h_m)\Big]}_{\text{total } U(x)}"
             r"= \underbrace{\tfrac{1}{M}\textstyle\sum_m H\big[p(y\mid x,h_m)\big]}_{\text{aleatoric } AU(x)}"
             r"+ \underbrace{I(y;h\mid x,D)}_{\text{epistemic } EU(x)}")
    st.latex(r"\mathrm{Var}[y\mid x] = \underbrace{\tfrac{1}{M}\textstyle\sum_m \sigma_m^2(x)}_{\text{aleatoric}}"
             r"+ \underbrace{\tfrac{1}{M}\textstyle\sum_m \big(\mu_m(x)-\bar\mu(x)\big)^2}_{\text{epistemic}}")
