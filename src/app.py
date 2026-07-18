from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="World Cup Predictor", layout="centered")

@st.cache_resource
def load_data():
    model = joblib.load(BASE_DIR / "model.pkl")
    le = joblib.load(BASE_DIR / "label_encoder.pkl")
    team_names = pd.read_csv(BASE_DIR / "../data/raw/team_names.csv", index_col="id")["name"]
    elo = pd.read_csv(BASE_DIR / "../data/raw/latest_elo.csv", index_col="team_id")
    form = pd.read_csv(BASE_DIR / "../data/raw/latest_form.csv", index_col="team_id")
    h2h = pd.read_csv(BASE_DIR / "../data/raw/latest_h2h.csv")
    return model, le, team_names, elo, form, h2h

model, le, team_names, elo, form, h2h = load_data()

def get_h2h(home_id, away_id):
    a, b = min(home_id, away_id), max(home_id, away_id)
    match = h2h[(h2h["team_a"] == a) & (h2h["team_b"] == b)]

    if len(match) == 0:
        return 0.5, 0

    row = match.iloc[0]
    rate = row["h2h_home_win_rate"]
    if row["home_team_id"] != home_id:
        rate = 1 - rate

    return rate, row["h2h_matches_played"]

def build_features(home_id, away_id):
    home_elo = elo.loc[home_id, "elo"] if home_id in elo.index else 1500
    away_elo = elo.loc[away_id, "elo"] if away_id in elo.index else 1500
    home_form = form.loc[home_id, "form_points"] if home_id in form.index else 0
    away_form = form.loc[away_id, "form_points"] if away_id in form.index else 0
    h2h_rate, h2h_played = get_h2h(home_id, away_id)

    return pd.DataFrame([{
        "home_form": home_form,
        "away_form": away_form,
        "h2h_home_win_rate": h2h_rate,
        "h2h_matches_played": h2h_played,
        "home_elo": home_elo,
        "away_elo": away_elo
    }])

st.title("World Cup match predictor")
st.caption("Built on national-team fixtures, qualifiers, and continental championships — not just World Cup matches.")

team_options = team_names.sort_values().to_dict()
name_to_id = {v: k for k, v in team_options.items()}
team_list = list(name_to_id.keys())

col1, col2 = st.columns(2)
with col1:
    home_name = st.selectbox("Home team", team_list, index=0)
with col2:
    away_name = st.selectbox("Away team", team_list, index=1)

if st.button("Predict", type="primary"):
    if home_name == away_name:
        st.warning("Pick two different teams.")
    else:
        home_id = name_to_id[home_name]
        away_id = name_to_id[away_name]

        X = build_features(home_id, away_id)
        probs = model.predict_proba(X)[0]
        classes = le.inverse_transform(range(len(probs)))
        prob_dict = dict(zip(classes, probs))

        st.subheader("Win probability")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(home_name, f"{prob_dict.get('home_win', 0):.0%}")
        with c2:
            st.metric("Draw", f"{prob_dict.get('draw', 0):.0%}")
        with c3:
            st.metric(away_name, f"{prob_dict.get('away_win', 0):.0%}")

        st.divider()

        outcome_labels = {"home_win": home_name, "draw": "Draw", "away_win": away_name}
        predicted = max(prob_dict, key=prob_dict.get)
        st.info(f"Most likely outcome: **{outcome_labels[predicted]}** ({prob_dict[predicted]:.0%})")
        # to run it use streamlit run src/app.py