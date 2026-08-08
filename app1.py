import streamlit as st
import pickle
import re
import pandas as pd
import altair as alt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Load Saved Models & Data
# -----------------------------
clf = pickle.load(open("disease_model.pkl", "rb"))
tfidf_clf = pickle.load(open("tfidf.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))

clinical = pd.read_csv("clinical_day2_clusters.csv")

# -----------------------------
# NLP Preprocessing
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_disease(text):
    cleaned = clean_text(text)
    vector = tfidf_clf.transform([cleaned])
    prediction = clf.predict(vector)
    return label_encoder.inverse_transform(prediction)[0]


# -----------------------------
# Similarity Function
# -----------------------------
def get_similar_trials(index, n=5):
    trial_vector = vectorizer.transform([clinical.iloc[index]["clean_summary"]])
    scores = cosine_similarity(
        trial_vector, vectorizer.transform(clinical["clean_summary"])
    )[0]
    scores = list(enumerate(scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in scores[1 : n + 1]]
    results = clinical.iloc[top_indices][
        ["Disease", "clean_summary", "Cluster_Label"]
    ].copy()
    results["Similarity_Score"] = [s for _, s in scores[1 : n + 1]]
    return results


# -----------------------------
# Streamlit Layout & Styling
# -----------------------------
st.set_page_config(page_title="Supercool Medical Chatbot Dashboard", layout="wide")

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #e3f2fd, #fce4ec);
    }
    .stButton>button {
        background: linear-gradient(90deg, #42a5f5, #478ed1);
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: bold;
    }
    .card {
        padding:20px;
        border-radius:12px;
        text-align:center;
        box-shadow:2px 2px 8px rgba(0,0,0,0.2);
        margin:10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("💊 Clinical Trial Disease Intelligence Assistant")

st.markdown("""
Welcome to your interactive medical AI assistant.  
Here you can:
- **Predict diseases** from trial summaries
- **Explore clusters** of clinical trials
- **Find similar trials** with recommendations
""")

# -----------------------------
# Tabs for Navigation
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["🔮 Disease Prediction", "📊 Cluster Exploration", "🧩 Recommendations"]
)

# --- Tab 1: Disease Prediction ---
with tab1:
    st.header("Disease Category Prediction")
    user_input = st.text_area("Enter Clinical Trial Summary")
    if st.button("Predict Disease"):
        if user_input.strip() != "":
            result = predict_disease(user_input)
            st.markdown(
                f"""
                <div class='card' style="background:linear-gradient(135deg,#81C784,#388E3C);color:white;">
                    <h3>Predicted Disease</h3>
                    <h2>{result}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Please enter a clinical trial summary.")

# --- Tab 2: Cluster Exploration ---
with tab2:
    st.header("Cluster Stats")

    # Sidebar filter (only cluster select box)
    cluster_filter = st.sidebar.selectbox(
        "Select Cluster", clinical["Cluster_Label"].unique()
    )

    # Apply filter
    filtered = clinical[clinical["Cluster_Label"] == cluster_filter]

    # Colorful metric cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='card' style='background:linear-gradient(135deg,#9C27B0,#E1BEE7);color:white;'><h4>Total Trials</h4><h2>{len(clinical)}</h2></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='card' style='background:linear-gradient(135deg,#009688,#4DB6AC);color:white;'><h4>Clusters</h4><h2>{clinical['Cluster_Label'].nunique()}</h2></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='card' style='background:linear-gradient(135deg,#FF7043,#FFCC80);color:white;'><h4>Diseases</h4><h2>{clinical['Disease'].nunique()}</h2></div>",
            unsafe_allow_html=True,
        )

    # Trial Summaries
    st.subheader("📋 Trial Summaries")
    for i, row in filtered.head(5).iterrows():
        with st.expander(f"🔍 {row['Disease']} (Cluster: {row['Cluster_Label']})"):
            st.write(row["clean_summary"])
            if st.button(f"Show Similar Trials for Index {i}"):
                similar = get_similar_trials(i, n=5)
                st.write(similar)

    # Charts in tabs
    tabA, tabB = st.tabs(["Cluster Distribution", "PCA Visualization"])
    with tabA:
        cluster_counts = clinical["Cluster_Label"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster_Label", "Count"]
        chart = (
            alt.Chart(cluster_counts)
            .mark_bar()
            .encode(
                x=alt.X("Cluster_Label", sort="-y"),
                y="Count",
                color=alt.Color("Cluster_Label", scale=alt.Scale(scheme="pastel1")),
            )
            .properties(width=700, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    with tabB:
        X_text = vectorizer.transform(clinical["clean_summary"])
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X_text.toarray())
        pca_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        pca_df["Cluster_Label"] = clinical["Cluster_Label"]
        scatter_chart = (
            alt.Chart(pca_df)
            .mark_circle(size=60)
            .encode(
                x="PC1",
                y="PC2",
                color=alt.Color("Cluster_Label", scale=alt.Scale(scheme="set2")),
                tooltip=["Cluster_Label"],
            )
            .interactive()
        )
        st.altair_chart(scatter_chart, use_container_width=True)

# --- Tab 3: Recommendations ---
with tab3:
    st.header("Find Similar Trials")

    # User enters a disease name instead of index
    disease_name = st.text_input("Enter Disease Name")

    if st.button("Show Similar Trials"):
        if disease_name.strip() != "":
            match = clinical[clinical["Disease"].str.contains(disease_name, case=False)]
            if not match.empty:
                index = match.index[0]  # take the first match
                similar = get_similar_trials(index, n=5)
                st.write(similar)
            else:
                st.warning("No trials found for that disease name.")
        else:
            st.warning("Please enter a disease name.")
