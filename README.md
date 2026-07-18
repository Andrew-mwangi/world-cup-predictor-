# ⚽ World Cup Match Predictor

Predicting international football match outcomes using team form, Elo-style strength ratings, and head-to-head history — built end-to-end from raw API data to a live app.

**[Try it live →](#)** *(link goes here once deployed)*

---

## Why I built this

I was watching a lot of football during this World Cup season and noticed everyone — pundits, fans, betting sites — predicts matches off gut feeling. I wanted to see if I could do it more systematically: pull real match data, engineer proper features, and let a model calculate probabilities instead of guessing.

The catch: the World Cup itself only has 64 matches every four years — nowhere near enough to train on. So the real challenge wasn't the modeling, it was figuring out how to responsibly expand the dataset without losing what makes international football different from club football.

## How it works

**Data:** Pulled ~1,700 real matches via the [API-Football](https://www.api-football.com/) API — not just World Cup matches, but qualifiers, continental championships (Euros, AFCON, Copa América), UEFA Nations League, and friendlies. All national-team fixtures are kept consistent in context.

**Features, calculated with zero data leakage:**
- **Recent form** — points from each team's last 5 matches, using only results *before* the match being predicted
- **Elo rating** — a chess-style running strength score that updates after every match based on who beat whom, and how surprising the result was
- **Head-to-head history** — win rate between the two specific teams, when enough history exists to be meaningful

**Model:** Gradient boosting classifier (scikit-learn's `HistGradientBoostingClassifier`), trained on a chronological train/test split — tested only on matches that happened after the training data, to simulate genuinely predicting the future rather than the past.

**Frontend:** Streamlit app — pick two teams, get a win/draw/loss probability breakdown along with the underlying stats driving the prediction.

## Results, honestly reported

- **~60% accuracy** on held-out matches, vs. a **46% baseline** (always guessing the home team wins)
- Strong at distinguishing wins and losses; **draws remain the hardest outcome to call** — consistent with how difficult draws are to predict in football generally, even for human pundits
- Every feature addition was tested against a real baseline before being kept — a couple of promising-looking features (goal difference specifically) were tested, and **dropped** after being shown to add noise rather than signal, rather than kept just because they sounded useful

## Tech stack

`Python` · `pandas` · `scikit-learn` · `Streamlit` · `API-Football`

## Project structure

```
world-cup-predictor/
├── data/raw/          # Pulled and processed match data
├── notebooks/         # Exploration and feature-engineering work
├── src/
│   ├── fetch_data.py
│   ├── build_features.py
│   ├── train_model.py
│   └── app.py         # Streamlit app
└── requirements.txt
```

## Running it locally

```bash
git clone https://github.com/Andrew-mwangi/world-cup-predictor-
cd world-cup-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

You'll need your own free API-Football key in a `.env` file:
```
API_FOOTBALL_KEY=your_key_here
```

## What's next

Squad and player-level data (injuries, key absences) — deliberately scoped out of v1 since official lineups aren't confirmed until ~1 hour before kickoff, which needs careful handling rather than a quick bolt-on
Improving draw prediction, specifically, once richer features are in place

## About

Built by Andrew Maina, interested in applying AI and data to real-world problems.

---

*This is an educational/portfolio project, not betting advice.*
