import streamlit as st
import pandas as pd
import datetime
import hashlib
import os

# Initialize CSV files if they don't exist
def init_excel_files():
    files = {
        'users.csv': ['username', 'password', 'workout_config'],
        'workouts.csv': ['username', 'Date', 'Exercise', 'Set', 'Reps', 'Weight'],
        'weights.csv': ['username', 'Date', 'Weight']
    }
    for file, columns in files.items():
        if not os.path.exists(file):
            pd.DataFrame(columns=columns).to_csv(file, index=False)

# Default workout configuration
DEFAULT_WORKOUT_CONFIG = {
    'Monday': ['Bench Press', 'Incline Press', 'Chest Flyes', 'Tricep Extensions'],
    'Tuesday': ['Deadlifts', 'Pull-ups', 'Barbell Rows', 'Bicep Curls'],
    'Wednesday': ['Squats', 'Leg Press', 'Leg Extensions', 'Calf Raises'],
    'Thursday': ['Shoulder Press', 'Lateral Raises', 'Front Raises', 'Shrugs'],
    'Friday': ['Bench Press', 'Pull-ups', 'Shoulder Press', 'Arms Superset'],
    'Saturday': ['Full Body Workout'],
    'Sunday': ['Rest Day']
}

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists('users.csv'):
        return pd.read_csv('users.csv')
    return pd.DataFrame(columns=['username', 'password', 'workout_config'])

def save_user(username, password, workout_config):
    users_df = load_users()
    new_user = pd.DataFrame([[username, hash_password(password), str(workout_config)]], 
                           columns=['username', 'password', 'workout_config'])
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv('users.csv', index=False)

def authenticate(username, password):
    users_df = load_users()
    user = users_df[users_df['username'] == username]
    if not user.empty:
        if user.iloc[0]['password'] == hash_password(password):
            return True
    return False

# Data handling functions
def load_workout_data(username):
    if os.path.exists('workouts.csv'):
        df = pd.read_csv('workouts.csv')
        return df[df['username'] == username]
    return pd.DataFrame(columns=['username', 'Date', 'Exercise', 'Set', 'Reps', 'Weight'])

def save_workout_data(df, username):
    all_df = pd.read_csv('workouts.csv')
    all_df = all_df[all_df['username'] != username]
    df['username'] = username
    final_df = pd.concat([all_df, df], ignore_index=True)
    final_df.to_csv('workouts.csv', index=False)

def load_weight_data(username):
    if os.path.exists('weights.csv'):
        df = pd.read_csv('weights.csv')
        return df[df['username'] == username]
    return pd.DataFrame(columns=['username', 'Date', 'Weight'])

def save_weight_data(df, username):
    all_df = pd.read_csv('weights.csv')
    all_df = all_df[all_df['username'] != username]
    df['username'] = username
    final_df = pd.concat([all_df, df], ignore_index=True)
    final_df.to_csv('weights.csv', index=False)

def get_user_workout_config(username):
    users_df = load_users()
    user = users_df[users_df['username'] == username]
    if not user.empty:
        return eval(user.iloc[0]['workout_config'])
    return DEFAULT_WORKOUT_CONFIG

def update_workout_config(username, new_config):
    users_df = load_users()
    users_df.loc[users_df['username'] == username, 'workout_config'] = str(new_config)
    users_df.to_csv('users.csv', index=False)

# Page functions
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🏋️ Gym Tracker Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", use_container_width=True):
                if authenticate(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        with col2:
            if st.button("Sign Up", use_container_width=True):
                st.session_state['show_signup'] = True
                st.rerun()

def signup_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Create Account")
        new_username = st.text_input("Choose Username")
        new_password = st.text_input("Choose Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back to Login", use_container_width=True):
                st.session_state['show_signup'] = False
                st.rerun()
        with col2:
            if st.button("Create Account", use_container_width=True):
                if new_password != confirm_password:
                    st.error("Passwords don't match!")
                elif not new_username or not new_password:
                    st.error("Please fill all fields!")
                else:
                    users_df = load_users()
                    if new_username in users_df['username'].values:
                        st.error("Username already exists!")
                    else:
                        save_user(new_username, new_password, DEFAULT_WORKOUT_CONFIG)
                        st.success("Account created successfully!")
                        st.session_state['show_signup'] = False
                        st.rerun()

def main_app():
    username = st.session_state['username']
    workout_config = get_user_workout_config(username)
    
    # Sidebar
    with st.sidebar:
        st.title(f"Welcome, {username}!")
        page = st.radio("Navigation", ["Workout Tracker", "Profile", "Logout"])
        
        if page == "Logout":
            st.session_state['logged_in'] = False
            st.rerun()
    
    if page == "Profile":
        st.title("Profile Settings")
        st.subheader("Customize Your Workout Schedule")
        
        new_config = {}
        for day in workout_config.keys():
            st.write(f"\n{day}")
            exercises = st.text_area(
                f"Exercises for {day}", 
                value='\n'.join(workout_config[day]),
                key=f"exercises_{day}"
            )
            new_config[day] = [ex.strip() for ex in exercises.split('\n') if ex.strip()]
        
        if st.button("Save Changes", use_container_width=True):
            update_workout_config(username, new_config)
            st.success("Workout schedule updated!")
    
    else:  # Workout Tracker
        st.title("🏋️ Gym Reps & Weight Tracker")
        date_str = str(datetime.date.today())
        today = datetime.datetime.today().strftime('%A')
        
        tab1, tab2 = st.tabs(["Workout Tracker", "Weight Tracker"])
        
        with tab1:
            st.subheader(f"Today's Workout Plan ({today})")
            
            # Input form
            col1, col2 = st.columns(2)
            with col1:
                exercise = st.selectbox("Exercise:", workout_config[today])
                reps = st.number_input("Reps:", min_value=1, step=1)
            with col2:
                set_number = st.number_input("Set:", min_value=1, step=1)
                weight = st.number_input("Weight (kg):", min_value=0.0, step=0.5)
            
            # Add entry button
            if st.button("Add Entry", use_container_width=True):
                df = load_workout_data(username)
                new_entry = pd.DataFrame([[username, date_str, exercise, set_number, reps, weight]], 
                                      columns=['username', 'Date', 'Exercise', 'Set', 'Reps', 'Weight'])
                df = pd.concat([df, new_entry], ignore_index=True)
                save_workout_data(df, username)
                st.success(f"Added: {exercise} - Set {set_number}, {reps} reps @ {weight}kg")
                st.rerun()
            
            # Show today's workout
            df = load_workout_data(username)
            today_workout = df[df['Date'] == date_str]
            if not today_workout.empty:
                st.subheader("Today's Progress")
                st.dataframe(today_workout[['Exercise', 'Set', 'Reps', 'Weight']])
        
        with tab2:
            st.subheader("Weight Tracker")
            
            col1, col2 = st.columns(2)
            with col1:
                weight = st.number_input("Your Weight (kg):", min_value=0.0, step=0.1)
            with col2:
                if st.button("Save Weight", use_container_width=True):
                    df_weight = load_weight_data(username)
                    new_entry = pd.DataFrame([[username, date_str, weight]], 
                                          columns=['username', 'Date', 'Weight'])
                    df_weight = pd.concat([df_weight, new_entry], ignore_index=True)
                    save_weight_data(df_weight, username)
                    st.success(f"Weight recorded: {weight}kg")
                    st.rerun()
            
            # Show weight history
            df_weight = load_weight_data(username)
            if not df_weight.empty:
                st.line_chart(df_weight.set_index('Date')['Weight'])

def main():
    init_excel_files()
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'show_signup' not in st.session_state:
        st.session_state['show_signup'] = False
    
    if not st.session_state['logged_in']:
        if st.session_state['show_signup']:
            signup_page()
        else:
            login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
