import streamlit as st
import pandas as pd
import pickle

# 1. Load the trained model pipeline
with open('ipl_win_predictor_pipeline.pkl', 'rb') as f:
    production_pipe = pickle.load(f)

st.title("🏏 IPL Live Win Predictor")
st.markdown("Enter the live match situation to predict the chasing team's win probability.")

# 2. Recreate the team and city lists used during training
teams = ['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bengaluru', 
         'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings', 
         'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants']

cities = ['Mumbai', 'Chennai', 'Bangalore', 'Kolkata', 'Delhi', 'Hyderabad', 'Ahmedabad', 'Jaipur']

# 3. Build UI Layout columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Chasing Team (Batting First)', sorted(teams))
with col2:
    # Ensure bowling team isn't the same as the batting team
    bowling_team = st.selectbox('Select Bowling Team', sorted([t for t in teams if t != batting_team]))

city = st.selectbox('Select Match Venue City', sorted(cities))

col3, col4, col5 = st.columns(3)
with col3:
    target = st.number_input('Target Score', min_value=1, max_value=300, value=180)
with col4:
    score = st.number_input('Current Score', min_value=0, max_value=300, value=120)
with col5:
    wickets_out = st.number_input('Wickets Out', min_value=0, max_value=9, value=4)

overs_completed = st.slider('Overs Completed', min_value=0.0, max_value=19.5, value=15.0, step=0.1)

# 4. Run the calculation logic on button click
if st.button('Predict Win Probability'):
    # Basic data handling mimicking your original helper function
    completed_overs_int = int(overs_completed)
    balls_in_current_over = round((overs_completed - completed_overs_int) * 10)
    legal_balls_bowled = completed_overs_int * 6 + balls_in_current_over
    
    runs_left = target - score
    balls_left = 120 - legal_balls_bowled
    wickets_left = 10 - wickets_out
    
    crr = (score * 6) / legal_balls_bowled if legal_balls_bowled > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    
    # Build feature DataFrame matching X[feature_cols]
    row = pd.DataFrame([{
        'batting_team': batting_team, 'bowling_team': bowling_team, 'city': city,
        'runs_left': runs_left, 'balls_left': balls_left, 'wickets_left': wickets_left,
        'target': target, 'crr': crr, 'rrr': rrr
    }])
    
    # Get ML Prediction
    win_prob = production_pipe.predict_proba(row)[0, 1]
    
    # Display Results
    st.subheader("Match Outcome Probability")
    st.success(f"**{batting_team}**: {win_prob * 100:.1f}% chance to win")
    st.error(f"**{bowling_team}**: {(1 - win_prob) * 100:.1f}% chance to win")