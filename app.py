import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide",
)

# ── Global styles (v1 aesthetic, green palette) ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --primary:   #346739;
    --secondary: #79AE6F;
    --tertiary:  #9FCB98;
    --accent:    #F2EDC2;
    --bg:        #0b110c;
    --bg2:       #101a11;
    --bg3:       #151f16;
    --border:    #1e2e1f;
    --border2:   #263827;
    --muted:     #5a7a5d;
    --text:      #e8ede8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background-color: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1200px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f1f10 0%, #122014 50%, #1a3320 100%);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '🌸';
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.12;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: var(--accent);
    margin: 0 0 0.4rem;
    letter-spacing: -1px;
}
.hero p {
    font-size: 1rem;
    color: var(--muted);
    margin: 0;
    font-weight: 300;
}

/* ── Cards ── */
.card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.section-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1rem;
    display: block;
}

/* ── Prediction result box ── */
.result-box {
    background: #0d2212;
    border: 1px solid var(--primary);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.result-box .species-name {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: var(--tertiary);
    display: block;
    margin-bottom: 0.3rem;
}
.result-box .result-label {
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.6rem;
}
.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #8b9e8d;
    width: 90px;
    flex-shrink: 0;
    text-transform: capitalize;
}
.prob-bar-bg {
    flex: 1;
    background: var(--border);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
}
.prob-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Dataset / expander ── */
.dataset-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.dataset-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: var(--accent);
}
.dataset-badge {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 99px;
    padding: 0.25rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--secondary);
}

/* ── Sliders ── */
.stSlider > div > div > div { background: var(--secondary) !important; }
div[data-testid="stSlider"] label {
    font-size: 0.82rem !important;
    color: #8b9e8d !important;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Metric boxes ── */
[data-testid="stMetricValue"] {
    color: var(--tertiary) !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── Caption ── */
.stCaption { color: var(--muted) !important; font-size: 0.78rem !important; }

/* ── Subheader ── */
h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--accent) !important;
    letter-spacing: -0.5px !important;
}

/* ── DataFrame ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model & encoder ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("iris_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# ── Load dataset for visuals ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("iris.csv")

df = load_data()

# ── Species colours (green palette) ──────────────────────────────────────────
COLORS = {"setosa": "#346739", "versicolor": "#79AE6F", "virginica": "#9FCB98"}
SPECIES_EMOJI = {"setosa": "🌼", "versicolor": "🌿", "virginica": "🌺"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg",
    use_container_width=True
)
st.sidebar.markdown(
    "<div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#F2EDC2;margin:0.5rem 0 0.2rem;'>🌸 Iris Classifier</div>"
    "<div style='font-size:0.8rem;color:#5a7a5d;margin-bottom:1rem;'>Adjust the sliders to predict the Iris species.</div>",
    unsafe_allow_html=True
)

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width  = st.sidebar.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

# ── Prediction ────────────────────────────────────────────────────────────────
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
pred_idx   = model.predict(input_data)[0]
pred_proba = model.predict_proba(input_data)[0]
pred_label = le.inverse_transform([pred_idx])[0]
color      = COLORS[pred_label]
emoji      = SPECIES_EMOJI.get(pred_label, "🌸")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>Iris Species Predictor</h1>
    <p>Explore the dataset, adjust measurements, and let the model identify the species in real time.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Top row: prediction card + probability bar ────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(
        f"""
        <div class='result-box' style='border-color:{color}; background:{color}18;'>
            <div style='font-size:48px; margin-bottom:4px;'>{emoji}</div>
            <span class='species-name' style='color:{color};'>Iris {pred_label.capitalize()}</span>
            <span class='result-label'>Predicted Species</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("<span class='section-label'>Prediction Confidence</span>", unsafe_allow_html=True)
    classes = le.classes_
    for sp, prob in zip(classes, pred_proba):
        pct = int(prob * 100)
        sp_color = COLORS[sp]
        sp_emoji = SPECIES_EMOJI.get(sp, "")
        st.markdown(f"""
        <div class='prob-row'>
            <span class='prob-label'>{sp_emoji} {sp}</span>
            <div class='prob-bar-bg'>
                <div class='prob-bar-fill' style='width:{pct}%; background:{sp_color};'></div>
            </div>
            <span class='prob-val'>{pct}%</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Second row: PCA scatter + feature importance ──────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Dataset — PCA Projection")
    pca = PCA(n_components=2)
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    X_2d = pca.fit_transform(X)
    user_2d = pca.transform(input_data)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor("#101a11")
    ax2.set_facecolor("#101a11")
    for species, grp in df.groupby("species"):
        idx = df.index[df["species"] == species]
        ax2.scatter(X_2d[idx, 0], X_2d[idx, 1],
                    color=COLORS[species], label=species.capitalize(),
                    alpha=0.75, edgecolors="none", s=50)
    ax2.scatter(user_2d[0, 0], user_2d[0, 1],
                color="#F2EDC2", s=220, marker="*", zorder=5,
                edgecolors=color, linewidths=1.2, label="Your input")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=9, color="#5a7a5d")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=9, color="#5a7a5d")
    ax2.tick_params(colors="#5a7a5d", labelsize=9)
    ax2.legend(fontsize=9, framealpha=0, labelcolor="#e8ede8")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color("#1e2e1f")
    st.pyplot(fig2)

with col4:
    st.subheader("Feature Importances")
    features    = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    importances = model.feature_importances_
    order       = np.argsort(importances)

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    fig3.patch.set_facecolor("#101a11")
    ax3.set_facecolor("#101a11")
    bar_colors = [COLORS["setosa"], COLORS["versicolor"], COLORS["virginica"], "#F2EDC2"]
    ax3.barh([features[i] for i in order], importances[order],
             color=[bar_colors[i % len(bar_colors)] for i in range(len(order))],
             edgecolor="none", height=0.5)
    ax3.set_xlabel("Importance", fontsize=9, color="#5a7a5d")
    ax3.spines[["top", "right", "bottom"]].set_visible(False)
    ax3.spines["left"].set_color("#1e2e1f")
    ax3.tick_params(labelsize=10, colors="#e8ede8")
    ax3.xaxis.label.set_color("#5a7a5d")
    st.pyplot(fig3)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Dataset preview ───────────────────────────────────────────────────────────
st.markdown(f"""
<div class='dataset-header'>
    <span class='dataset-title'>📋 Iris Dataset Reference</span>
    <span class='dataset-badge'>150 samples · 3 species · 4 features</span>
</div>
""", unsafe_allow_html=True)

with st.expander("View Full Dataset"):
    st.dataframe(df, use_container_width=True, height=260)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total samples", len(df))
    c2.metric("Features", 4)
    c3.metric("Classes", df["species"].nunique())

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Model: Random Forest (100 trees) · Trained on Iris dataset · Accuracy: 100%")
