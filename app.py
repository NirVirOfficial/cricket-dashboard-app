import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("match_data.csv")

df = load_data()

# Sidebar
st.sidebar.header("Match Configuration")
team1 = st.sidebar.selectbox("Select Team 1", df['batting_team'].unique())
team2 = st.sidebar.selectbox("Select Team 2", df['batting_team'].unique())
venue = st.sidebar.multiselect("Select Venue(s)", df['venue'].unique())
innings_filter = st.sidebar.radio("Batting First", ['Both', 'Team 1', 'Team 2'])

# Prevent comparing the same team
if team1 == team2:
    st.error("Please select two different teams for comparison.")
    st.stop()

# Filter data
def filter_data(team_a, team_b):
    matches = df[(df['batting_team'].isin([team_a, team_b])) &
                 (df['bowling_team'].isin([team_a, team_b]))]
    if venue:
        matches = matches[matches['venue'].isin(venue)]
    if innings_filter != 'Both':
        target_team = team_a if innings_filter == 'Team 1' else team_b
        first_inn_matches = matches[(matches['innings'] == 1) &
                                    (matches['batting_team'] == target_team)]['match_id']
        matches = matches[matches['match_id'].isin(first_inn_matches)]
    return matches

filtered_df = filter_data(team1, team2)

# Head-to-Head Analysis
st.header("🏏 Head-to-Head Analysis")

def calculate_wins(data, team_a, team_b):
    matches = data.drop_duplicates('match_id', keep='first')[['match_id', 'batting_team', 'bowling_team', 'runs_off_bat']]
    match_results = []

    for match_id in matches['match_id'].unique():
        match_data = data[data['match_id'] == match_id]
        first_inn = match_data[match_data['innings'] == 1]
        second_inn = match_data[match_data['innings'] == 2]

        if len(first_inn) == 0 or len(second_inn) == 0:
            continue

        team1_runs = first_inn['runs_off_bat'].sum() + first_inn['extras'].sum()
        team2_runs = second_inn['runs_off_bat'].sum() + second_inn['extras'].sum()

        winner = first_inn['batting_team'].iloc[0] if team1_runs > team2_runs else second_inn['batting_team'].iloc[0]
        match_results.append({'match_id': match_id, 'winner': winner})

    results_df = pd.DataFrame(match_results)
    if 'winner' not in results_df.columns:
        return 0, 0, 0

    team1_wins = results_df[results_df['winner'] == team1].shape[0]
    team2_wins = results_df[results_df['winner'] == team2].shape[0]
    total = team1_wins + team2_wins

    return total, team1_wins, team2_wins

total_matches, team1_wins, team2_wins = calculate_wins(filtered_df, team1, team2)
col1, col2, col3 = st.columns(3)
col1.metric("Total Matches", total_matches)
col2.metric(f"{team1} Wins", team1_wins)
col3.metric(f"{team2} Wins", team2_wins)

# Top Performers
st.header("⭐ Top Performers")

def get_top_performers(data, team):
    batsmen = data[data['batting_team'] == team].groupby('striker')['runs_off_bat'].sum().nlargest(5)
    bowlers = data[data['bowling_team'] == team].groupby('bowler')['wicket_type'].count().nlargest(5)
    return batsmen, bowlers

team1_batsmen, team1_bowlers = get_top_performers(filtered_df, team1)
team2_batsmen, team2_bowlers = get_top_performers(filtered_df, team2)

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"{team1} Top Batsmen")
    st.dataframe(team1_batsmen, use_container_width=True)
    st.subheader(f"{team1} Top Bowlers")
    st.dataframe(team1_bowlers, use_container_width=True)

with col2:
    st.subheader(f"{team2} Top Batsmen")
    st.dataframe(team2_batsmen, use_container_width=True)
    st.subheader(f"{team2} Top Bowlers")
    st.dataframe(team2_bowlers, use_container_width=True)

# Advanced Stats
st.header("📊 Advanced Statistics")
tab1, tab2, tab3 = st.tabs(["Run Distribution", "Wicket Types", "Venue Analysis"])

with tab1:
    fig = px.histogram(filtered_df, x='runs_off_bat', nbins=20, title="Run Distribution Histogram")
    st.plotly_chart(fig)

with tab2:
    wicket_data = filtered_df[filtered_df['wicket_type'].notnull()]
    fig = px.pie(wicket_data, names='wicket_type', title="Wicket Type Distribution")
    st.plotly_chart(fig)

with tab3:
    venue_stats = filtered_df.groupby('venue').agg({'runs_off_bat': 'sum', 'wicket_type': 'count'})
    fig = px.bar(venue_stats, x=venue_stats.index, y=['runs_off_bat', 'wicket_type'], title="Venue-wise Performance")
    st.plotly_chart(fig)

# Dream Team
st.header("👑 Combined Dream Team")
def create_dream_team():
    all_players = pd.concat([
        filtered_df['striker'],
        filtered_df['non_striker'],
        filtered_df['bowler']
    ]).value_counts().head(11)
    return all_players.index.tolist()

dream_team = create_dream_team()
st.write("Best 11 Players from Both Teams:")
st.write(", ".join(dream_team))

# Prediction Model
st.header("🔮 Match Prediction")
def predict_winner():
    try:
        prediction_df = filtered_df.copy()
        prediction_df['wides'] = prediction_df['wides'].fillna(0)
        prediction_df['extras'] = prediction_df['extras'].fillna(0)
        prediction_df['runs_off_bat'] = prediction_df['runs_off_bat'].fillna(0)

        match_stats = prediction_df.groupby('match_id').agg({
            'runs_off_bat': 'sum',
            'extras': 'sum',
            'wides': 'sum',
            'batting_team': 'first'
        }).reset_index()

        if len(match_stats) < 2:
            return "Not enough data", 0

        X = match_stats[['runs_off_bat', 'extras', 'wides']]
        y = match_stats['batting_team']

        model = LogisticRegression()
        model.fit(X, y)

        recent_data = match_stats.tail(5)[['runs_off_bat', 'extras', 'wides']].mean().values.reshape(1, -1)
        prediction = model.predict(recent_data)[0]
        proba = model.predict_proba(recent_data).max()

        return prediction, proba

    except Exception as e:
        return "Prediction unavailable", 0

predicted_winner, confidence = predict_winner()
if predicted_winner != "Prediction unavailable":
    st.metric("Predicted Winner", f"{predicted_winner} ({confidence * 100:.1f}% confidence)")
else:
    st.warning("Not enough data for prediction")
