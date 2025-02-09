import streamlit as st
import pandas as pd
import datetime
import hashlib
import os

def get_today_workout():
    schedule = {
        "Monday": ["Pull-ups", "Bicep Curl", "Hammer Curl"],
        "Tuesday": ["Incline Dumbbell Press", "Flat Dumbbell Press", "Weighted Dips", "Tricep Pushdown"],
        "Wednesday": ["Lateral Raises", "Shoulder Press", "Barbell Shrugs", "Rear Delts", "Hanging Leg Raises"],
        "Thursday": ["Pull-ups", "Bicep Curl", "Hammer Curl"],
        "Friday": ["Incline Dumbbell Press", "Flat Dumbbell Press", "Weighted Dips", "Tricep Pushdown"],
        "Saturday": ["Lateral Raises", "Shoulder Press", "Barbell Shrugs", "Rear Delts", "Hanging Leg Raises", "Squats", "Leg Curls", "Calf Raises"],
        "Sunday": ["rest"]
    }
    # test_day=st.text_input("enter day for test")
    # if test_day is None:
    #     test_day="Monday"
    actual=datetime.datetime.today().strftime("%A")
    return schedule[actual]

# File setup
WORKOUT_FILE = "workout_data.csv"
if not os.path.exists(WORKOUT_FILE):
    df = pd.DataFrame(columns=["Date", "Exercise", "Set", "Reps", "Weight"])
    df.to_csv(WORKOUT_FILE, index=False)

WEIGHT_FILE = "weight_data.csv"
if not os.path.exists(WEIGHT_FILE):
    df = pd.DataFrame(columns=["Date", "Weight"])
    df.to_csv(WEIGHT_FILE, index=False)

def load_workout_data():
    return pd.read_csv(WORKOUT_FILE)

def save_workout_data(df):
    df.to_csv(WORKOUT_FILE, index=False)

def load_weight_data():
    return pd.read_csv(WEIGHT_FILE)

def save_weight_data(df):
    df.to_csv(WEIGHT_FILE, index=False)

def main():
    st.set_page_config(page_title="Gym Reps Tracker", layout="wide")
    st.markdown(
        """
        <style>
            .stApp {
                background-image: url('https://source.unsplash.com/1600x900/?gym,fitness');
                background-size: cover;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🏋️ Gym Reps & Weight Tracker")
    
    date_str = str(datetime.date.today())
    
    tab1, tab2, tab3 = st.tabs(["Workout Tracker", "Weight Tracker", "Past Workouts"])
    
    with tab1:
        st.subheader(f"Today's Workout Plan ({datetime.datetime.today().strftime('%A')})")
        today_exercises = get_today_workout()
        
        df = load_workout_data()
        
        # Create 2 columns for exercise input
        col1, col2 = st.columns(2)
        with col1:
            exercise = st.selectbox("Select Exercise:", today_exercises)
            reps = st.number_input("Enter Number of Reps:", min_value=1, step=1)
        with col2:
            set_number = st.number_input("Set Number:", min_value=1, step=1)
            weight = st.number_input("Enter Weight Used (kg):", min_value=0.0, step=0.5)
        
        # Center the Add Entry button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Add Entry", use_container_width=True):
                new_entry = pd.DataFrame([[date_str, exercise, set_number, reps, weight]], 
                                      columns=["Date", "Exercise", "Set", "Reps", "Weight"])
                df = pd.concat([df, new_entry], ignore_index=True)
                save_workout_data(df)
                st.success(f"Added Set {set_number}: {reps} reps for {exercise} with {weight} kg")
                st.rerun()
        
        st.subheader("Workout History")
        filtered_df = df[df["Date"] == date_str]
        st.dataframe(filtered_df)
        
        # Center the Delete button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Delete Today's Data", use_container_width=True):
                df = df[df["Date"] != date_str]
                save_workout_data(df)
                st.success("Deleted all records for today!")
                st.rerun()
    
    with tab2:
        df_weight = load_weight_data()
        
        # Create columns for weight input
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            weight = st.number_input("Enter Your Weight (kg):", min_value=0.0, step=0.1)
            if st.button("Save Weight", use_container_width=True):
                df_weight = df_weight[df_weight["Date"] != date_str]
                new_entry = pd.DataFrame([[date_str, weight]], columns=["Date", "Weight"])
                df_weight = pd.concat([df_weight, new_entry], ignore_index=True)
                save_weight_data(df_weight)
                st.success(f"Weight recorded: {weight} kg")
                st.rerun()
        
        st.subheader("Weight History")
        st.dataframe(df_weight)
        
        # Center the Delete button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Delete Today's Weight", use_container_width=True):
                df_weight = df_weight[df_weight["Date"] != date_str]
                save_weight_data(df_weight)
                st.success("Deleted today's weight record!")
                st.rerun()
    
    with tab3:
        st.subheader("View Past Workouts")
        selected_date = st.date_input("Select Date to View:")
        past_df = df[df["Date"] == str(selected_date)]
        st.dataframe(past_df)
    
if __name__ == "__main__":
    main()
