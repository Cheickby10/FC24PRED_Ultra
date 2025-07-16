import streamlit as st
import pandas as pd
from model import load_models, predict_match
from teams import TEAMS
import os

st.set_page_config(page_title="FC24PRED", layout="wide")
st.title("⚽ FC24PRED - Prédiction IA des matchs FIFA 4x4 - Premier League")

# ---------- Chargement des données ----------
if not os.path.exists("results.csv"):
    pd.DataFrame(columns=["team1","team2","score1_ht","score2_ht","score1_ft","score2_ft"]).to_csv("results.csv", index=False)
data = pd.read_csv("results.csv")
models = load_models(data)

# ---------- Formulaire d'ajout de match ----------
st.header("📥 Ajouter un match")

with st.form("match_form"):
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Équipe 1", TEAMS, key="add_team1")
    with col2:
        team2 = st.selectbox("Équipe 2", [t for t in TEAMS if t != team1], key="add_team2")

    col3, col4 = st.columns(2)
    with col3:
        score1_ht = st.number_input(f"Buts de {team1} à la mi-temps", min_value=0, max_value=10, value=0)
        score1_ft = st.number_input(f"Buts de {team1} à la fin", min_value=0, max_value=10, value=0)
    with col4:
        score2_ht = st.number_input(f"Buts de {team2} à la mi-temps", min_value=0, max_value=10, value=0)
        score2_ft = st.number_input(f"Buts de {team2} à la fin", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("✅ Ajouter ce match")
    if submitted:
        new_row = pd.DataFrame([[team1, team2, score1_ht, score2_ht, score1_ft, score2_ft]],
                               columns=["team1","team2","score1_ht","score2_ht","score1_ft","score2_ft"])
        data = pd.concat([data, new_row], ignore_index=True)
        data.to_csv("results.csv", index=False)
        st.success("✅ Match ajouté avec succès !")

# ---------- Section prédiction ----------
st.header("🔮 Prédire un match")
col5, col6 = st.columns(2)
with col5:
    pred_team1 = st.selectbox("Sélectionnez l'équipe 1", TEAMS, key="pred_team1")
with col6:
    pred_team2 = st.selectbox("Sélectionnez l'équipe 2", [t for t in TEAMS if t != pred_team1], key="pred_team2")

if st.button("Lancer la prédiction"):
    pred = predict_match(pred_team1, pred_team2, data, models)
    st.subheader("📊 Résultat de la prédiction IA")
    st.markdown(f"**Score à la mi-temps** : {pred['ht_score']}")
    st.markdown(f"**Score final** : {pred['ft_score']}")
    st.markdown(f"**Issue probable du match** : **{pred['issue']}** ({pred['issue_proba']}%)")

    st.subheader(f"📈 5 derniers matchs de {pred_team1}")
    st.dataframe(pred["last5_team1"])

    st.subheader(f"📈 5 derniers matchs de {pred_team2}")
    st.dataframe(pred["last5_team2"])

    st.subheader("📊 Comparaison statistique")
    st.bar_chart(pred["comparison"])