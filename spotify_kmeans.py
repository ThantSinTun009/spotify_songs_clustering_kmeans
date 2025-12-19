import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import os

# ---------------------
# Load data & pipeline
# ---------------------
final_df = pd.read_csv("final_df.csv")

with open("kmeans_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)   

numerical_features = [
    "danceability","energy", "acousticness", "popularity", "duration_ms", "liveness", "loudness", "speechiness", "tempo", "valence"
]
categorical_features = ["mode", "key", "time_signature"]

# mode mapping (training used 0/1)
mode_map = {"Minor": 0, "Major": 1}

st.title("Spotify Songs Clustering (K-Means)")

# ---------------------
# Sidebar
# ---------------------

logo_path = "images/parami.jpg"

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown("**Name:** Thant Sin Tun")
st.sidebar.markdown("**Student ID:** PIUS20230003")

st.sidebar.markdown("---")  

st.sidebar.header("Parameter Interface")
x_feature = st.sidebar.selectbox("X-axis feature", numerical_features, index=0)
y_feature = st.sidebar.selectbox("Y-axis feature", numerical_features, index=1)
k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)

# ---------------------
# Preprocess + PCA
# ---------------------
X_preprocessed = loaded_model.named_steps["preprocessor"].transform(final_df)
X_pca = loaded_model.named_steps["pca"].transform(X_preprocessed)

# ---------------------
# Dynamic KMeans 
# ---------------------
kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
kmeans.fit(X_pca)

labels = kmeans.labels_
df_plot = final_df.copy()
df_plot["cluster"] = labels

# ---------------------
# Plot: Original feature space
# ---------------------
st.subheader("Clusters in Original Feature Space")

fig, ax = plt.subplots()
for c in range(k):
    data = df_plot[df_plot["cluster"] == c]
    ax.scatter(
        data[x_feature],
        data[y_feature],
        alpha=0.6,
        edgecolors="#0e1111",
        label=f"Cluster {c}",
    )

ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.legend()
st.pyplot(fig)

# ---------------------
# Plot: PCA space
# ---------------------
st.subheader("Clusters in PCA Space")

fig2, ax2 = plt.subplots()
for c in range(k):
    ax2.scatter(
        X_pca[labels == c, 0],
        X_pca[labels == c, 1],
        alpha=0.6,
        edgecolors="#0e1111",
        label=f"Cluster {c}",
    )

# Centroids
centroids = kmeans.cluster_centers_
ax2.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    color="white",
    edgecolors="black",
    label="Centroids",
)

ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.legend()
st.pyplot(fig2)

# ---------------------
# Predict new song
# ---------------------
st.subheader("Predict Cluster for New Song")

c1, c2 = st.columns(2)
c3, c4 = st.columns(2)
c5, c6 = st.columns(2)
c7, c8 = st.columns(2)
c9, c10 = st.columns(2)
c11, c12, c13 = st.columns(3)

with c1: popularity = st.slider("Popularity", 0.0, 100.0, 30.0)
with c2: acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)
with c3: danceability = st.slider("Danceability", 0.0, 1.0, 0.3)
with c4: duration_ms = st.slider("Duration (ms)", 0.0, 600000.0, 180000.0)
with c5: energy = st.slider("Energy", 0.0, 1.0, 0.5)
with c6: liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
with c7: loudness = st.slider("Loudness (dB)", -60.0, 0.0, -35.0)
with c8: speechiness = st.slider("Speechiness", 0.0, 1.0, 0.7)
with c9: tempo = st.slider("Tempo (BPM)", 0.0, 250.0, 120.0)
with c10: valence = st.slider("Valence", 0.0, 1.0, 0.3)
with c11: key = st.selectbox("Key", sorted(final_df["key"].unique()))
with c12: mode = st.selectbox("Mode", ["Minor", "Major"])
with c13: time_signature = st.selectbox(
    "Time Signature", sorted(final_df["time_signature"].unique())
)

# ---------------------
# Predict button
# ---------------------
if st.button("Predict Cluster"):
    new_song = pd.DataFrame(
        [[
            danceability, acousticness, popularity, duration_ms,
            energy, liveness, loudness, speechiness, tempo, valence,
            mode_map[mode], key, time_signature
        ]],
        columns=numerical_features + categorical_features,
    )

    X_new = loaded_model.named_steps["preprocessor"].transform(new_song)
    X_new_pca = loaded_model.named_steps["pca"].transform(X_new)

    predicted_values = kmeans.predict(X_new_pca)
    st.success(f"This song belongs to **Cluster {predicted_values[0]}**")

