import streamlit as st
import pandas as pd
from datetime import datetime, date
import hashlib
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json
import logging
import traceback
import base64
from PIL import Image
import io
import glob

# Load environment variables
load_dotenv()

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Setup logging
def setup_logger():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a logger
    logger = logging.getLogger('GymTracker')
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    # File handler for all logs
    file_handler = logging.FileHandler(
        f'logs/gym_tracker_{date.today().strftime("%Y%m%d")}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for error logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

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

def save_user(username, password, workout_config, user_info):
    logger.info(f"Creating new user account for username: {username}")
    try:
        users_df = load_users()
        new_user = pd.DataFrame([[username, hash_password(password), str(workout_config), str(user_info)]], 
                               columns=['username', 'password', 'workout_config', 'user_info'])
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv('users.csv', index=False)
        logger.info(f"Successfully created user account for: {username}")
    except Exception as e:
        logger.error(f"Error creating user account: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def authenticate(username, password):
    logger.info(f"Attempting authentication for username: {username}")
    try:
        users_df = load_users()
        user = users_df[users_df['username'] == username]
        if not user.empty:
            if user.iloc[0]['password'] == hash_password(password):
                logger.info(f"Successful authentication for username: {username}")
                return True
        logger.warning(f"Failed authentication attempt for username: {username}")
        return False
    except Exception as e:
        logger.error(f"Error during authentication: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Data handling functions
def load_workout_data(username):
    if os.path.exists('workouts.csv'):
        df = pd.read_csv('workouts.csv')
        return df[df['username'] == username]
    return pd.DataFrame(columns=['username', 'Date', 'Exercise', 'Set', 'Reps', 'Weight'])

def save_workout_data(df, username):
    logger.info(f"Saving workout data for user: {username}")
    try:
        all_df = pd.read_csv('workouts.csv')
        all_df = all_df[all_df['username'] != username]
        df['username'] = username
        final_df = pd.concat([all_df, df], ignore_index=True)
        final_df.to_csv('workouts.csv', index=False)
        logger.info(f"Successfully saved workout data for user: {username}")
    except Exception as e:
        logger.error(f"Error saving workout data: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

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
    logger.info(f"Updating workout config for user: {username}")
    try:
        users_df = load_users()
        users_df.loc[users_df['username'] == username, 'workout_config'] = str(new_config)
        users_df.to_csv('users.csv', index=False)
        logger.info(f"Successfully updated workout config for user: {username}")
    except Exception as e:
        logger.error(f"Error updating workout config: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def get_user_info(username):
    logger.info(f"Getting user info for: {username}")
    try:
        users_df = load_users()
        user = users_df[users_df['username'] == username]
        if not user.empty:
            user_info = user.iloc[0]['user_info']
            logger.debug(f"Raw user_info from DB: {user_info}")
            # Handle the case where user_info might be nan or empty
            if pd.isna(user_info) or user_info == '':
                logger.warning(f"No user info found for {username}, returning default")
                return {}
            try:
                # Convert string representation of dict to actual dict
                return eval(user_info) if isinstance(user_info, str) else user_info
            except:
                logger.error(f"Error parsing user_info: {user_info}")
                return {}
        logger.warning(f"User {username} not found")
        return {}
    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}

def update_user_info(username, new_info):
    logger.info(f"Attempting to update user info for: {username}")
    logger.debug(f"New info to be updated: {new_info}")
    
    try:
        # Load current data
        users_df = load_users()
        if username not in users_df['username'].values:
            logger.error(f"Username {username} not found in database")
            raise ValueError(f"User {username} not found")
        
        # Get current user data
        current_data = users_df[users_df['username'] == username].iloc[0].to_dict()
        logger.debug(f"Current user data: {current_data}")
        
        # Update user_info
        users_df.loc[users_df['username'] == username, 'user_info'] = str(new_info)
        
        # Verify the update
        updated_data = users_df[users_df['username'] == username].iloc[0].to_dict()
        logger.debug(f"Updated user data: {updated_data}")
        
        # Save to CSV
        users_df.to_csv('users.csv', index=False)
        logger.info(f"Successfully updated user info for: {username}")
        
        return True
    except Exception as e:
        logger.error(f"Error updating user info: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def get_ai_workout_plan(user_info):
    logger.info(f"Generating AI workout plan for user info: {user_info}")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", 
                                google_api_key=GOOGLE_API_KEY,
                                temperature=0.7)
    
    # Get available days
    days_per_week = user_info['days_per_week']
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    available_days = days_of_week[:days_per_week]
    rest_days = days_of_week[days_per_week:]
    
    logger.debug(f"Available days: {available_days}")
    logger.debug(f"Rest days: {rest_days}")
    
    prompt = f"""
    Create a personalized {days_per_week}-day workout plan based on the following user information:
    - Weight: {user_info['weight']} kg
    - Height: {user_info['height']} cm
    - Fitness Goal: {user_info['goal']}
    - Experience Level: {user_info['experience']}
    - Available Days: {days_per_week} days per week

    Available workout days: {', '.join(available_days)}
    Rest days: {', '.join(rest_days)}

    Please provide a structured workout plan with:
    1. Exercises only for the {days_per_week} available days
    2. 3-5 exercises per muscle group
    3. Consider their experience level
    4. Optimize the split based on available days
    5 if user is available for 6 days a week, then the plan should be for 6 days a week
    6 plan should be equal to number of days available

    Return the response in this exact JSON format, including ONLY the specified available days and rest days:
    {{
        "Monday": ["Exercise 1", "Exercise 2", "Exercise 3"],
        // ... only include the actual available days and rest days
    }}

    For {days_per_week} days, focus on:
    - Proper muscle group split
    - Progressive overload
    - Adequate rest between similar muscle groups
    - {user_info['goal']} oriented exercises

    Only return the JSON object, no additional text.
    """
    
    try:
        logger.debug("Sending prompt to Gemini AI")
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        logger.debug(f"Received response: {response_text}")
        
        workout_plan = json.loads(response_text)
        
        # Validate the workout plan has correct number of days
        if len(workout_plan) != 7:
            # Add rest days to complete the week
            for day in rest_days:
                workout_plan[day] = ["Rest Day"]
        
        logger.info("Successfully generated and parsed workout plan")
        logger.debug(f"Final workout plan: {workout_plan}")
        return workout_plan
    except Exception as e:
        logger.error(f"Error generating workout plan: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        st.error(f"Error generating workout plan: {str(e)}")
        return DEFAULT_WORKOUT_CONFIG

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
        
        # Account Information
        st.subheader("Account Information")
        new_username = st.text_input("Choose Username")
        new_password = st.text_input("Choose Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        # Personal Information
        st.subheader("Personal Information")
        weight = st.number_input("Your Current Weight (kg)", min_value=0.0, step=0.1)
        height = st.number_input("Your Height (cm)", min_value=0.0, step=0.1)
        
        # Fitness Goals
        st.subheader("Fitness Goals")
        goal = st.selectbox("Your Primary Goal", 
                          ["Weight Loss", "Muscle Gain", "Strength Training", 
                           "General Fitness", "Body Recomposition"])
        
        experience = st.selectbox("Gym Experience",
                                ["Beginner (0-6 months)",
                                 "Intermediate (6-18 months)",
                                 "Advanced (18+ months)"])
        
        days_per_week = st.slider("How many days per week do you plan to workout?",
                                min_value=1, max_value=7, value=4)
        
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
                elif weight <= 0 or height <= 0:
                    st.error("Please enter valid weight and height!")
                else:
                    users_df = load_users()
                    if new_username in users_df['username'].values:
                        st.error("Username already exists!")
                    else:
                        # Create user info dictionary
                        user_info = {
                            'weight': weight,
                            'height': height,
                            'goal': goal,
                            'experience': experience,
                            'days_per_week': days_per_week,
                            'join_date': str(date.today())
                        }
                        save_user(new_username, new_password, DEFAULT_WORKOUT_CONFIG, user_info)
                        st.success("Account created successfully!")
                        st.session_state['show_signup'] = False
                        st.rerun()

def main_app():
    try:
        username = st.session_state['username']
        workout_config = get_user_workout_config(username)
        user_info = get_user_info(username)
        
        logger.info(f"Loading profile for user: {username}")
        
        with st.sidebar:
            st.title(f"Welcome, {username}!")
            page = st.radio("Navigation", ["Workout Tracker", "Profile", "Logout"])
            
            if page == "Logout":
                st.session_state['logged_in'] = False
                st.rerun()
        
        if page == "Profile":
            try:
                st.title("Profile Settings")
                
                # Personal Information Section
                st.subheader("Personal Information")
                col1, col2 = st.columns(2)
                with col1:
                    new_weight = st.number_input("Weight (kg)", 
                                               value=float(user_info.get('weight', 0)),
                                               step=0.1)
                    new_goal = st.selectbox("Primary Goal", 
                                          ["Weight Loss", "Muscle Gain", "Strength Training", 
                                           "General Fitness", "Body Recomposition"],
                                          index=["Weight Loss", "Muscle Gain", "Strength Training", 
                                                "General Fitness", "Body Recomposition"].index(user_info.get('goal', 'General Fitness')))
                with col2:
                    new_height = st.number_input("Height (cm)", 
                                               value=float(user_info.get('height', 0)),
                                               step=0.1)
                    new_experience = st.selectbox("Gym Experience",
                                                ["Beginner (0-6 months)",
                                                 "Intermediate (6-18 months)",
                                                 "Advanced (18+ months)"],
                                                index=["Beginner (0-6 months)",
                                                      "Intermediate (6-18 months)",
                                                      "Advanced (18+ months)"].index(user_info.get('experience', 'Beginner (0-6 months)')))
                
                new_days = st.slider("Workout Days per Week",
                                   min_value=1, max_value=7,
                                   value=user_info.get('days_per_week', 4))
                
                if new_height > 0:
                    bmi = new_weight / ((new_height/100) ** 2)
                    st.info(f"Your BMI: {bmi:.1f}")
                
                st.info(f"Member since: {user_info.get('join_date', 'N/A')}")
                
                # Update personal information
                if st.button("Update Personal Information", use_container_width=True):
                    try:
                        logger.info(f"Updating personal information for user: {username}")
                        
                        new_info = {
                            'weight': float(new_weight),
                            'height': float(new_height),
                            'goal': str(new_goal),
                            'experience': str(new_experience),
                            'days_per_week': int(new_days),
                            'join_date': user_info.get('join_date', str(date.today()))
                        }
                        logger.debug(f"Prepared new info: {new_info}")
                            
                        
                        
                        # Validate the data
                        if new_weight <= 0 or new_height <= 0:
                            raise ValueError("Weight and height must be greater than 0")
                        
                        update_success = update_user_info(username, new_info)
                        if update_success:
                            st.success("Personal information updated successfully!")
                            logger.info(f"Successfully updated personal information for user: {username}")
                            # Force a refresh of the page
                            st.rerun()
                        else:
                            st.error("Failed to update information. Please try again.")
                    except ValueError as ve:
                        logger.error(f"Validation error: {str(ve)}")
                        st.error(f"Invalid data: {str(ve)}")
                    except Exception as e:
                        logger.error(f"Error updating personal information: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        st.error("Failed to update personal information. Please check the logs for details.")
                
                # AI Workout Plan Generation
                st.subheader("AI Workout Plan Generator")
                st.info("Generate a personalized workout plan based on your profile using AI")
                
                if st.button("Generate AI Workout Plan", use_container_width=True):
                    try:
                        with st.spinner("Generating your personalized workout plan..."):
                            new_workout_plan = get_ai_workout_plan(user_info)
                            update_workout_config(username, new_workout_plan)
                            st.success("New workout plan generated and saved!")
                            st.rerun()
                    except Exception as e:
                        logger.error(f"Error generating AI workout plan: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        st.error("Failed to generate workout plan. Please try again.")
                
                # Display Current Workout Schedule
                st.subheader("Current Workout Schedule")
                st.info("You can either use the AI-generated plan or customize it manually below")
                
                try:
                    new_config = {}
                    for day in workout_config.keys():
                        st.write(f"\n{day}")
                        exercises = st.text_area(
                            f"Exercises for {day}", 
                            value='\n'.join(workout_config[day]),
                            key=f"exercises_{day}"
                        )
                        new_config[day] = [ex.strip() for ex in exercises.split('\n') if ex.strip()]
                    
                    if st.button("Save Custom Workout Plan", use_container_width=True):
                        update_workout_config(username, new_config)
                        st.success("Workout schedule updated successfully!")
                except Exception as e:
                    logger.error(f"Error displaying/updating workout schedule: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    st.error("Error updating workout schedule. Please try again.")
                    
            except Exception as e:
                logger.error(f"Error in profile page: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                st.error("Error loading profile page. Please try again.")
        
        else:
            st.title("🏋️ Gym Reps & Weight Tracker")
            date_str = str(date.today())
            today = datetime.today().strftime('%A')
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Workout Tracker", 
                "Weight Tracker", 
                "Progress Pictures",
                "Analytics",
                "AI Assistant"
            ])
            
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
                        try:
                            # Save to weight tracker
                            df_weight = load_weight_data(username)
                            new_entry = pd.DataFrame([[username, date_str, weight]], 
                                                  columns=['username', 'Date', 'Weight'])
                            df_weight = pd.concat([df_weight, new_entry], ignore_index=True)
                            save_weight_data(df_weight, username)
                            
                            # Update user profile weight
                            user_info = get_user_info(username)
                            user_info['weight'] = float(weight)
                            update_user_info(username, user_info)
                            
                            st.success(f"Weight recorded: {weight}kg and profile updated!")
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Error saving weight: {str(e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            st.error("Failed to save weight. Please try again.")
                
                # Show weight history
                df_weight = load_weight_data(username)
                if not df_weight.empty:
                    st.line_chart(df_weight.set_index('Date')['Weight'])
            
            with tab3:
                st.subheader("📸 Progress Pictures")
                
                # Create two columns for upload options
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Upload from File")
                    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
                    if uploaded_file is not None:
                        # Display preview
                        st.image(uploaded_file, caption="Preview", use_column_width=True)
                        if st.button("Save File Upload"):
                            if save_uploaded_image(username, uploaded_file, date_str, "file"):
                                st.success("Progress picture saved successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to save progress picture")
                
                with col2:
                    st.subheader("Take Picture")
                    camera_photo = st.camera_input("Take a picture")
                    if camera_photo is not None:
                        # Display preview
                        st.image(camera_photo, caption="Preview", use_column_width=True)
                        if st.button("Save Camera Photo"):
                            if save_uploaded_image(username, camera_photo, date_str, "camera"):
                                st.success("Progress picture saved successfully!")
                                st.rerun()
                
                # Display progress pictures gallery
                st.subheader("Your Progress Gallery")
                images = get_user_images(username)
                
                if images:
                    # Create a date filter
                    dates = [os.path.basename(img).split('_')[0] for img in images]
                    unique_dates = sorted(list(set(dates)), reverse=True)
                    selected_date = st.selectbox("Filter by date:", unique_dates)
                    
                    # Filter images by selected date
                    filtered_images = [img for img in images if selected_date in img]
                    
                    # Display images in a grid
                    cols = st.columns(3)
                    for idx, image_path in enumerate(filtered_images):
                        # Extract date and source from filename
                        filename = os.path.basename(image_path)
                        date_str, source = filename.replace('.jpg', '').split('_')
                        
                        with cols[idx % 3]:
                            st.image(image_path, caption=f"Date: {date_str}\nSource: {source}")
                            if st.button(f"Delete", key=f"delete_{idx}"):
                                try:
                                    os.remove(image_path)
                                    st.success("Image deleted successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to delete image: {str(e)}")
                    
                    if (idx + 1) % 3 == 0:  # Create new row after every 3 images
                        cols = st.columns(3)
                else:
                    st.info("No progress pictures uploaded yet")

            with tab4:
                st.subheader("📊 Fitness Analytics Dashboard")
                
                # Get user's data
                workout_data = load_workout_data(username)
                weight_data = load_weight_data(username)
                
                # Check if data exists
                if workout_data.empty and weight_data.empty:
                    st.warning("No workout or weight data available yet. Start logging your workouts and weight to see your analytics!")
                else:
                    # Create tabs for different analytics
                    analytics_tabs = st.tabs([
                        "Workout Progress",
                        "Weight Trends",
                        "Personal Records",
                        "Exercise Frequency",
                        "Goal Tracking"
                    ])
                    
                    # 1. Workout Progress
                    with analytics_tabs[0]:
                        st.subheader("💪 Workout Progress")
                        
                        if not workout_data.empty:
                            # Calculate total volume (weight * reps) per workout
                            workout_data['Volume'] = workout_data['Weight'] * workout_data['Reps']
                            
                            # Group by date and calculate metrics
                            daily_metrics = workout_data.groupby('Date').agg({
                                'Volume': 'sum',
                                'Set': 'count',
                                'Exercise': 'nunique'
                            }).reset_index()
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Workouts", len(daily_metrics))
                            with col2:
                                st.metric("Total Sets", daily_metrics['Set'].sum())
                            with col3:
                                st.metric("Total Volume", f"{daily_metrics['Volume'].sum():,.0f} kg")
                            
                            # Progress charts
                            st.subheader("Daily Training Volume")
                            st.line_chart(daily_metrics.set_index('Date')['Volume'])
                            
                            st.subheader("Sets per Workout")
                            st.line_chart(daily_metrics.set_index('Date')['Set'])
                        else:
                            st.info("Start logging your workouts to see progress analytics!")
                    
                    # 2. Weight Trends
                    with analytics_tabs[1]:
                        st.subheader("⚖️ Weight Trends")
                        
                        if not weight_data.empty:
                            # Calculate weight changes
                            initial_weight = weight_data['Weight'].iloc[0]
                            current_weight = weight_data['Weight'].iloc[-1]
                            weight_change = current_weight - initial_weight
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Starting Weight", f"{initial_weight:.1f} kg")
                            with col2:
                                st.metric("Current Weight", f"{current_weight:.1f} kg")
                            with col3:
                                st.metric("Weight Change", 
                                        f"{abs(weight_change):.1f} kg",
                                        delta=f"{weight_change:+.1f} kg")
                            
                            # Weight trend chart
                            st.subheader("Weight Progress")
                            st.line_chart(weight_data.set_index('Date')['Weight'])
                            
                            # Calculate statistics
                            if len(weight_data) > 1:
                                weekly_change = weight_change / (len(weight_data) / 7)
                                st.info(f"Average weekly change: {weekly_change:+.2f} kg")
                        else:
                            st.info("Start tracking your weight to see trends!")
                    
                    # 3. Personal Records
                    with analytics_tabs[2]:
                        st.subheader("🏆 Personal Records")
                        
                        if not workout_data.empty:
                            # Get max weight for each exercise
                            pr_data = workout_data.groupby('Exercise').agg({
                                'Weight': 'max',
                                'Reps': 'max',
                                'Date': 'last'
                            }).reset_index()
                            
                            pr_data.columns = ['Exercise', 'Max Weight (kg)', 'Max Reps', 'Last Performed']
                            st.dataframe(pr_data, hide_index=True)
                            
                            # Show PR history for selected exercise
                            st.subheader("PR Progress Chart")
                            selected_exercise = st.selectbox(
                                "Select Exercise",
                                options=workout_data['Exercise'].unique()
                            )
                            
                            exercise_progress = workout_data[workout_data['Exercise'] == selected_exercise]
                            exercise_prs = exercise_progress.groupby('Date')['Weight'].max()
                            st.line_chart(exercise_prs)
                        else:
                            st.info("Log your workouts to track your personal records!")
                    
                    # 4. Exercise Frequency
                    with analytics_tabs[3]:
                        st.subheader("📊 Exercise Frequency Analysis")
                        
                        if not workout_data.empty:
                            # Exercise frequency
                            exercise_freq = workout_data['Exercise'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Most Performed Exercises")
                                st.bar_chart(exercise_freq)
                            
                            with col2:
                                st.subheader("Exercise Distribution")
                                exercise_dist = workout_data['Exercise'].value_counts().head(10)
                                st.dataframe(exercise_dist.reset_index(), 
                                           columns=['Exercise', 'Times Performed'],
                                           hide_index=True)
                            
                            # Workout frequency calendar
                            st.subheader("Workout Calendar")
                            workout_dates = workout_data['Date'].unique()
                            workout_freq = len(workout_dates)
                            total_days = (pd.to_datetime(workout_data['Date'].max()) - 
                                        pd.to_datetime(workout_data['Date'].min())).days + 1
                            
                            consistency = (workout_freq / total_days) * 100
                            st.metric("Workout Consistency", 
                                    f"{consistency:.1f}%",
                                    help="Percentage of days with recorded workouts")
                        else:
                            st.info("Start logging workouts to see frequency analysis!")
                    
                    # 5. Goal Progress
                    with analytics_tabs[4]:
                        st.subheader("🎯 Goal Progress")
                        
                        # Get user's goal from profile
                        user_goal = user_info.get('goal', 'General Fitness')
                        st.write(f"Current Goal: **{user_goal}**")
                        
                        if user_goal == "Weight Loss":
                            if not weight_data.empty:
                                weight_change = current_weight - initial_weight
                                st.metric("Total Weight Loss", 
                                        f"{abs(weight_change):.1f} kg",
                                        delta=f"{-weight_change:.1f} kg")
                                
                                # BMI calculation if height is available
                                if 'height' in user_info:
                                    height_m = float(user_info['height']) / 100
                                    current_bmi = current_weight / (height_m ** 2)
                                    st.metric("Current BMI", f"{current_bmi:.1f}")
                        
                        elif user_goal in ["Muscle Gain", "Strength Training"]:
                            if not workout_data.empty:
                                # Show strength progression for key exercises
                                key_exercises = ['Bench Press', 'Squats', 'Deadlifts']
                                for exercise in key_exercises:
                                    if exercise in workout_data['Exercise'].values:
                                        st.write(f"**{exercise} Progress**")
                                        exercise_data = workout_data[workout_data['Exercise'] == exercise]
                                        max_weights = exercise_data.groupby('Date')['Weight'].max()
                                        st.line_chart(max_weights)
                        
                        elif user_goal == "General Fitness":
                            if not workout_data.empty:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Exercise Variety", 
                                            f"{len(exercise_freq)} exercises")
                                with col2:
                                    st.metric("Workout Consistency", 
                                            f"{consistency:.1f}%")
                        
                        # Overall progress summary
                        st.subheader("Progress Summary")
                        total_days = (datetime.now() - pd.to_datetime(user_info.get('join_date'))).days
                        st.info(f"You've been working out for {total_days} days!")

            with tab5:
                st.subheader("🤖 AI Fitness Assistant")
                
                # Add expandable section with example questions
                with st.expander("📝 What can you ask the AI Assistant?"):
                    st.markdown("""
                    You can ask questions about:
                    
                    **Progress Analysis**
                    - "How has my weight changed over the last month?"
                    - "Am I making good progress towards my fitness goals?"
                    - "What trends do you see in my workout data?"
                    
                    **Workout Plan**
                    - "Is my current workout plan aligned with my goals?"
                    - "Should I increase weights for any exercises?"
                    - "How can I modify my routine for better results?"
                    
                    **Personal Recommendations**
                    - "Based on my progress, what should I focus on next?"
                    - "How can I improve my workout consistency?"
                    - "What exercises would complement my current routine?"
                    
                    **General Fitness Advice**
                    - "How can I prevent plateaus in my training?"
                    - "What's the best way to track my progress?"
                    - "How can I optimize my rest days?"
                    """)

                st.info("Ask me anything about your fitness journey! I can help analyze your progress, suggest improvements, or answer questions about your workout plan.")

                # Initialize messages in session state if not present
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Add a clear chat button
                col1, col2 = st.columns([5,1])
                with col2:
                    if st.button("Clear Chat"):
                        st.session_state.messages = []
                        st.rerun()

                # Get all user data for context
                workout_data = load_workout_data(username)
                weight_data = load_weight_data(username)
                progress_pics = len(get_user_images(username))
                
                # Format the context for the AI
                context = f"""
                User Profile:
                - Weight: {user_info.get('weight', 'N/A')} kg
                - Height: {user_info.get('height', 'N/A')} cm
                - Fitness Goal: {user_info.get('goal', 'N/A')}
                - Experience Level: {user_info.get('experience', 'N/A')}
                - Member since: {user_info.get('join_date', 'N/A')}
                - Workout days per week: {user_info.get('days_per_week', 'N/A')}

                Current Workout Plan:
                {json.dumps(workout_config, indent=2)}

                Progress Data:
                - Total workouts recorded: {len(workout_data)}
                - Weight measurements: {len(weight_data)}
                - Progress pictures: {progress_pics}
                - Latest weight: {weight_data['Weight'].iloc[-1] if not weight_data.empty else 'N/A'} kg
                """

                # Display chat messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Accept user input
                if prompt := st.chat_input("Ask your fitness question..."):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Generate AI response
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-pro",
                            google_api_key=GOOGLE_API_KEY,
                            temperature=0.7
                        )

                        # Include chat history in the context
                        chat_history = "\n".join([
                            f"{msg['role']}: {msg['content']}" 
                            for msg in st.session_state.messages[-5:]  # Last 5 messages for context
                        ])

                        full_prompt = f"""
                        You are a knowledgeable fitness assistant with access to the user's complete fitness data.
                        Please provide a helpful response based on the following context and question.

                        User Context:
                        {context}

                        Recent Chat History:
                        {chat_history}

                        User Question: {prompt}

                        Please provide a detailed, personalized response using the available data.
                        Keep in mind the conversation history for context.
                        """

                        response = llm.invoke(full_prompt)
                        ai_response = response.content

                        # Display AI response
                        with st.chat_message("assistant"):
                            st.markdown(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        
                        # Force scroll to bottom
                        st.rerun()

                    except Exception as e:
                        logger.error(f"Error generating AI response: {str(e)}")
                        st.error("Sorry, I couldn't generate a response. Please try again.")

    except Exception as e:
        logger.error(f"Error in main_app: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        st.error("An unexpected error occurred. Please try again later.")

def main():
    logger.info("Starting application")
    try:
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
            
    except Exception as e:
        logger.error(f"Critical application error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        st.error("An unexpected error occurred. Please try again later.")

# Add new functions for image handling
def save_uploaded_image(username, image_data, date_str, source="file"):
    """Save uploaded image with date and username"""
    try:
        # Create user's image directory if it doesn't exist
        user_image_dir = f'user_progress_pics/{username}'
        if not os.path.exists(user_image_dir):
            os.makedirs(user_image_dir)
        
        # Convert to PIL Image if it's not already
        if isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data))
        else:
            img = Image.open(image_data)
        
        # Save the image with date and source in filename
        filename = f"{user_image_dir}/{date_str}_{source}.jpg"
        img.save(filename)
        logger.info(f"Saved progress picture for user {username} on {date_str} from {source}")
        return True
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def get_user_images(username):
    """Get all progress pictures for a user"""
    try:
        user_image_dir = f'user_progress_pics/{username}'
        if not os.path.exists(user_image_dir):
            return []
        
        # Get all jpg files and sort by date
        images = glob.glob(f"{user_image_dir}/*.jpg")
        images.sort(reverse=True)  # Most recent first
        return images
    except Exception as e:
        logger.error(f"Error getting user images: {str(e)}")
        return []

if __name__ == "__main__":
    main()
