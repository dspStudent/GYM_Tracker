import streamlit as st
import pandas as pd
from datetime import datetime, date
import hashlib
from db import get_db, get_users_collection, get_workouts_collection, get_weights_collection, get_workout_by_id, get_weight_by_id # Added get_weight_by_id
from bson import ObjectId # Import ObjectId

# Initialize DB connection
MONGO_CONNECTION_STRING = "mongodb+srv://dev:dev@cluster0.hwutjuq.mongodb.net/"

def get_db_connection():
    return get_db(MONGO_CONNECTION_STRING)
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

import pymongo

# Initialize Database (e.g., create indexes)
def init_db():
    logger.info("Initializing database and creating indexes if they don't exist.")
    try:
        db = get_db_connection()
        if db is not None:
            users_collection = get_users_collection(db)
            workouts_collection = get_workouts_collection(db)
            weights_collection = get_weights_collection(db)

            # Create index for users collection (username should be unique)
            users_collection.create_index([('username', pymongo.ASCENDING)], unique=True)
            logger.info("Created/ensured index on 'username' (unique) for 'users' collection.")

            # Create index for workouts collection
            workouts_collection.create_index([('username', pymongo.ASCENDING)])
            logger.info("Created/ensured index on 'username' for 'workouts' collection.")

            # Create index for weights collection
            weights_collection.create_index([('username', pymongo.ASCENDING)])
            logger.info("Created/ensured index on 'username' for 'weights' collection.")
            
            logger.info("Database initialization complete.")
        else:
            logger.error("Failed to connect to MongoDB. Database initialization skipped.")
    except pymongo.errors.PyMongoError as e:
        logger.error(f"MongoDB error during database initialization: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

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
    logger.info("Loading users from MongoDB.")
    try:
        db = get_db_connection()
        if db is not None:
            users_collection = get_users_collection(db)
            users_data = list(users_collection.find({}, {"_id": 0}))  # Exclude _id field
            if users_data:
                return pd.DataFrame(users_data)
        return pd.DataFrame(columns=['username', 'password', 'workout_config', 'user_info'])
    except Exception as e:
        logger.error(f"Error loading users from MongoDB: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(columns=['username', 'password', 'workout_config', 'user_info'])

def save_user(username, password, workout_config, user_info):
    logger.info(f"Creating new user account for username: {username}")
    try:
        db = get_db_connection()
        if db is not None:
            users_collection = get_users_collection(db)
            user_document = {
                'username': username,
                'password': hash_password(password),
                'workout_config': str(workout_config),
                'user_info': str(user_info)
            }
            users_collection.insert_one(user_document)
            logger.info(f"Successfully created user account for: {username} in MongoDB.")
        else:
            logger.error("Failed to connect to MongoDB. User not saved.")
            raise Exception("Failed to connect to MongoDB")
    except Exception as e:
        logger.error(f"Error creating user account in MongoDB: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def authenticate(username, password):
    logger.info(f"Attempting authentication for username: {username}")
    try:
        db = get_db_connection()
        if db is not None:
            users_collection = get_users_collection(db)
            user_data = users_collection.find_one({'username': username})
            if user_data and user_data['password'] == hash_password(password):
                logger.info(f"Successful authentication for username: {username}")
                return True
        logger.warning(f"Failed authentication attempt for username: {username}")
        return False
    except Exception as e:
        logger.error(f"Error during authentication with MongoDB: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Data handling functions
def load_workout_data(username):
    logger.info(f"Loading workout data for user: {username} from MongoDB.")
    try:
        db = get_db_connection()
        if db is not None:
            workouts_collection = get_workouts_collection(db)
            # Fetch all fields including _id
            workout_data = list(workouts_collection.find({'username': username})) 
            if workout_data:
                # Convert list of dicts to DataFrame
                df = pd.DataFrame(workout_data)
                # Ensure correct column order and handle missing columns
                # Add '_id' to expected columns if you want it in the DataFrame,
                # or handle it separately when iterating.
                # For this task, we'll ensure it's present for button keys.
                expected_columns = ['_id', 'username', 'Date', 'Exercise', 'Set', 'Reps', 'Weight']
                # Reindex, adding missing columns with NaN, and ensuring _id is present
                df = df.reindex(columns=expected_columns) 
                return df
        # Return empty DataFrame with correct columns if no data or DB connection fails
        return pd.DataFrame(columns=['_id', 'username', 'Date', 'Exercise', 'Set', 'Reps', 'Weight'])
    except Exception as e:
        logger.error(f"Error loading workout data from MongoDB: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(columns=['username', 'Date', 'Exercise', 'Set', 'Reps', 'Weight'])

def save_workout_data(df, username):
    logger.info(f"Saving workout data for user: {username} to MongoDB.")
    try:
        db = get_db_connection()
        if db is not None:
            workouts_collection = get_workouts_collection(db)
            # Delete existing workout data for the user
            delete_result = workouts_collection.delete_many({'username': username})
            logger.info(f"Deleted {delete_result.deleted_count} existing workout documents for user: {username}.")
            
            # Convert DataFrame to list of dictionaries for insertion
            # Ensure 'username' is set for all records in the DataFrame
            df['username'] = username 
            records_to_insert = df.to_dict('records')

            # Process records to remove invalid _id fields
            processed_records = []
            for record in records_to_insert:
                if '_id' in record and (pd.isna(record['_id']) or record['_id'] is None):
                    del record['_id']  # Let MongoDB generate _id for new entries
                processed_records.append(record)
            
            if processed_records:
                insert_result = workouts_collection.insert_many(processed_records)
                logger.info(f"Successfully inserted {len(insert_result.inserted_ids)} workout documents for user: {username}.")
            else:
                logger.info(f"No workout data to insert for user: {username}.")
        else:
            logger.error("Failed to connect to MongoDB. Workout data not saved.")
            raise Exception("Failed to connect to MongoDB")
    except Exception as e:
        logger.error(f"Error saving workout data to MongoDB: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def load_weight_data(username):
    logger.info(f"Loading weight data for user: {username} from MongoDB.")
    try:
        db = get_db_connection()
        if db is not None:
            weights_collection = get_weights_collection(db)
            # Fetch all fields including _id
            weight_data = list(weights_collection.find({'username': username}))
            if weight_data:
                # Convert list of dicts to DataFrame
                df = pd.DataFrame(weight_data)
                # Ensure correct column order and handle missing columns
                expected_columns = ['_id', 'username', 'Date', 'Weight']
                df = df.reindex(columns=expected_columns) # Add _id to expected columns
                return df
        # Return empty DataFrame with correct columns if no data or DB connection fails
        return pd.DataFrame(columns=['_id', 'username', 'Date', 'Weight'])
    except Exception as e:
        logger.error(f"Error loading weight data from MongoDB: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(columns=['username', 'Date', 'Weight'])

def save_weight_data(df, username):
    logger.info(f"Saving weight data for user: {username} to MongoDB.")
    try:
        db = get_db_connection()
        if db is not None:
            weights_collection = get_weights_collection(db)
            # Delete existing weight data for the user
            delete_result = weights_collection.delete_many({'username': username})
            logger.info(f"Deleted {delete_result.deleted_count} existing weight documents for user: {username}.")
            
            # Convert DataFrame to list of dictionaries for insertion
            # Ensure 'username' is set for all records in the DataFrame
            df['username'] = username
            records_to_insert = df.to_dict('records')

            # Process records to remove invalid _id fields
            processed_records = []
            for record in records_to_insert:
                if '_id' in record and (pd.isna(record['_id']) or record['_id'] is None):
                    del record['_id']  # Let MongoDB generate _id for new entries
                processed_records.append(record)
            
            if processed_records:
                insert_result = weights_collection.insert_many(processed_records)
                logger.info(f"Successfully inserted {len(insert_result.inserted_ids)} weight documents for user: {username}.")
            else:
                logger.info(f"No weight data to insert for user: {username}.")
        else:
            logger.error("Failed to connect to MongoDB. Weight data not saved.")
            raise Exception("Failed to connect to MongoDB")
    except Exception as e:
        logger.error(f"Error saving weight data to MongoDB: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def get_user_workout_config(username):
    logger.info(f"Getting workout config for user: {username} from MongoDB.")
    try:
        db = get_db_connection()
        if db is not None:
            users_collection = get_users_collection(db)
            user_data = users_collection.find_one({'username': username})
            if user_data and 'workout_config' in user_data:
                return eval(user_data['workout_config'])
        return DEFAULT_WORKOUT_CONFIG
    except Exception as e:
        logger.error(f"Error getting workout config from MongoDB: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return DEFAULT_WORKOUT_CONFIG

def update_workout_config(username, new_config):
    logger.info(f"Updating workout config for user: {username} in MongoDB.")
    try:
        db = get_db_connection()
        if db is not None:
            users_collection = get_users_collection(db)
            users_collection.update_one(
                {'username': username},
                {'$set': {'workout_config': str(new_config)}}
            )
            logger.info(f"Successfully updated workout config for user: {username} in MongoDB.")
        else:
            logger.error("Failed to connect to MongoDB. Workout config not updated.")
            raise Exception("Failed to connect to MongoDB")
    except Exception as e:
        logger.error(f"Error updating workout config in MongoDB: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def get_user_info(username):
    logger.info(f"Getting user info for: {username} from MongoDB.")
    try:
        db = get_db_connection()
        if db is not None:
            users_collection = get_users_collection(db)
            user_data = users_collection.find_one({'username': username})
            if user_data and 'user_info' in user_data:
                user_info_str = user_data['user_info']
                logger.debug(f"Raw user_info from MongoDB: {user_info_str}")
                if pd.isna(user_info_str) or user_info_str == '':
                    logger.warning(f"No user info found for {username} in MongoDB, returning default")
                    return {}
                try:
                    return eval(user_info_str) if isinstance(user_info_str, str) else user_info_str
                except Exception as parse_error:
                    logger.error(f"Error parsing user_info from MongoDB: {user_info_str} - {parse_error}")
                    return {}
        logger.warning(f"User {username} not found in MongoDB.")
        return {}
    except Exception as e:
        logger.error(f"Error getting user info from MongoDB: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}

def update_user_info(username, new_info):
    logger.info(f"Attempting to update user info for: {username} in MongoDB.")
    logger.debug(f"New info to be updated: {new_info}")
    
    try:
        db = get_db_connection()
        if db is not None:
            users_collection = get_users_collection(db)
            # Check if user exists
            if users_collection.find_one({'username': username}) is None:
                logger.error(f"Username {username} not found in MongoDB.")
                raise ValueError(f"User {username} not found")

            users_collection.update_one(
                {'username': username},
                {'$set': {'user_info': str(new_info)}}
            )
            logger.info(f"Successfully updated user info for: {username} in MongoDB.")
            return True
        else:
            logger.error("Failed to connect to MongoDB. User info not updated.")
            raise Exception("Failed to connect to MongoDB")
    except Exception as e:
        logger.error(f"Error updating user info in MongoDB: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Ensure init_excel_files is either removed or adapted if some CSVs are still needed.
# For now, we assume it's not needed for users.csv anymore.
# If workouts.csv and weights.csv are also moving to MongoDB, this function will need more changes.
def init_excel_files():
    files = {
        # 'users.csv': ['username', 'password', 'workout_config', 'user_info'], # Removed users.csv
        # 'workouts.csv': ['username', 'Date', 'Exercise', 'Set', 'Reps', 'Weight'], # Removed workouts.csv
        # 'weights.csv': ['username', 'Date', 'Weight'] # Removed weights.csv
    }
    for file, columns in files.items():
        if not os.path.exists(file):
            pd.DataFrame(columns=columns).to_csv(file, index=False)

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
            # Removed "Date Config" from sidebar navigation
            page = st.radio("Navigation", ["Workout Tracker", "Profile", "Config", "Logout"]) 
            
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

        elif page == "Config":
            st.title("⚙️ Config")
            st.subheader("Configure Log Date")

            # Ensure selected_date is initialized
            if 'selected_date' not in st.session_state:
                st.session_state.selected_date = date.today()

            new_date_config = st.date_input(
                "Select date for logging/viewing entries:",
                value=st.session_state.selected_date,
                max_value=date.today(), # Prevent future dates
                help="This date will be used for logging new entries and viewing past entries in trackers."
            )
            if new_date_config != st.session_state.selected_date:
                st.session_state.selected_date = new_date_config
                # Also update custom_log_date if it exists, to keep it in sync when not actively using "Log for a Different Date" mode
                if 'custom_log_date' in st.session_state:
                    st.session_state.custom_log_date = new_date_config
                # If setting a custom date, implicitly switch mode for clarity, or let tab-specific controls handle it.
                # For now, just setting selected_date. Tabs will use this.
                # st.session_state.date_config_mode = "Log for a Different Date" # Optional: auto-switch mode
                st.rerun()

            if st.button("Set Log Date to Today"):
                if st.session_state.selected_date != date.today():
                    st.session_state.selected_date = date.today()
                    if 'custom_log_date' in st.session_state: # Keep custom_log_date in sync
                        st.session_state.custom_log_date = date.today()
                    # st.session_state.date_config_mode = "Log for Today" # Optional: auto-switch mode
                    st.rerun()
                else:
                    st.info("Log date is already set to today.")
            
            st.info(f"Entries in 'Workout Tracker' and 'Weight Tracker' will now be for: {st.session_state.selected_date.strftime('%Y-%m-%d')}")
        
        else: # Default to Workout Tracker page (and other main tabs)
            # Initialize date-related session state variables if they don't exist
            if 'selected_date' not in st.session_state:
                st.session_state.selected_date = date.today()
            if 'date_config_mode' not in st.session_state:
                st.session_state.date_config_mode = "Log for Today"
            # custom_log_date might be needed later if we allow selecting a date for "Log for a Different Date" mode
            if 'custom_log_date' not in st.session_state:
                 st.session_state.custom_log_date = date.today()

            # Determine the current log date based on the mode
            if st.session_state.date_config_mode == "Log for Today":
                st.session_state.selected_date = date.today() # Ensure it's always today in this mode
            # If "Log for a Different Date", selected_date will be set by a date_input (to be added in next step)
            # For now, it will hold its previous value or today's date if just switched.

            current_log_date = st.session_state.selected_date
            date_str = str(current_log_date) 
            today = current_log_date.strftime('%A') 

            st.title("🏋️ Gym Reps & Weight Tracker")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Workout Tracker", 
                "Weight Tracker", 
                "Progress Pictures",
                "Analytics",
                "AI Assistant"
            ])
            
            with tab1:
                st.info(f"Displaying and logging entries for: {date_str}") # Added informational message
                st.subheader(f"Workout Plan for {today} ({date_str})") 

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
                df_all_workouts = load_workout_data(username) # Renamed to avoid confusion
                today_workout_entries = df_all_workouts[df_all_workouts['Date'] == date_str]

                if not today_workout_entries.empty:
                    st.subheader("Today's Progress")
                    for index, entry in today_workout_entries.iterrows(): # Use iterrows() for DataFrames
                        entry_id = str(entry['_id']) # Ensure _id is a string for keys
                        
                        col1, col2, col3, col4, col5, col6 = st.columns([3,1,1,1,1,1])
                        with col1:
                            st.write(f"{entry['Exercise']}")
                        with col2:
                            st.write(f"Set: {entry['Set']}")
                        with col3:
                            st.write(f"Reps: {entry['Reps']}")
                        with col4:
                            st.write(f"Wt: {entry['Weight']}")
                        with col5:
                            if st.button("Edit", key=f"edit_{entry_id}"):
                                st.session_state.editing_workout_id = entry_id
                                st.rerun()
                        with col6:
                            if st.button("Delete", key=f"delete_{entry_id}"):
                                st.session_state.deleting_workout_id = entry_id
                                st.rerun()
                else:
                    st.info("No workouts logged for today yet.")

                # Placeholder for edit/delete actions based on session state
                # Note: These placeholders were for debugging and might be removed or integrated into the forms later.
                # For now, they are commented out to avoid clutter as the forms themselves indicate the state.
                # if 'editing_workout_id' in st.session_state and st.session_state.editing_workout_id:
                #     st.write(f"Placeholder: Editing workout ID {st.session_state.editing_workout_id}")
                
                # if 'deleting_workout_id' in st.session_state and st.session_state.deleting_workout_id:
                #     st.write(f"Placeholder: Attempting to delete workout ID {st.session_state.deleting_workout_id}")
                
                # --- Edit Workout Form ---
                if 'editing_workout_id' in st.session_state and st.session_state.editing_workout_id:
                    workout_to_edit_id_str = st.session_state.editing_workout_id
                    db_conn = get_db_connection()
                    if db_conn is not None:
                        workout_to_edit = get_workout_by_id(db_conn, workout_to_edit_id_str)

                        if workout_to_edit:
                            st.subheader("Edit Workout Entry")
                            with st.form(key="edit_workout_form"):
                                # Assuming workout_config[today] is available and relevant
                                # If exercises can be custom, use st.text_input
                                current_exercise_index = 0 # Default
                                if workout_to_edit['Exercise'] in workout_config[today]:
                                    current_exercise_index = workout_config[today].index(workout_to_edit['Exercise'])
                                
                                edited_exercise = st.selectbox(
                                    "Exercise:", 
                                    options=workout_config[today], 
                                    index=current_exercise_index,
                                    key=f"edit_exercise_{workout_to_edit_id_str}"
                                )
                                edited_set = st.number_input(
                                    "Set:", 
                                    value=int(workout_to_edit['Set']), 
                                    min_value=1, 
                                    step=1,
                                    key=f"edit_set_{workout_to_edit_id_str}"
                                )
                                edited_reps = st.number_input(
                                    "Reps:", 
                                    value=int(workout_to_edit['Reps']), 
                                    min_value=1, 
                                    step=1,
                                    key=f"edit_reps_{workout_to_edit_id_str}"
                                )
                                edited_weight = st.number_input(
                                    "Weight (kg):", 
                                    value=float(workout_to_edit['Weight']), 
                                    min_value=0.0, 
                                    step=0.5,
                                    key=f"edit_weight_{workout_to_edit_id_str}"
                                )

                                save_button = st.form_submit_button("Save Changes")
                                cancel_button = st.form_submit_button("Cancel")

                                if save_button:
                                    try:
                                        workouts_collection = get_workouts_collection(db_conn)
                                        query_id = ObjectId(workout_to_edit_id_str)
                                        
                                        update_data = {
                                            "$set": {
                                                "Exercise": edited_exercise,
                                                "Set": edited_set,
                                                "Reps": edited_reps,
                                                "Weight": edited_weight
                                            }
                                        }
                                        result = workouts_collection.update_one({"_id": query_id}, update_data)
                                        
                                        if result.modified_count > 0:
                                            st.success("Workout updated successfully!")
                                        else:
                                            st.warning("No changes made or workout not found.")
                                        st.session_state.editing_workout_id = None
                                        st.rerun()
                                    except Exception as e:
                                        logger.error(f"Error updating workout: {e}")
                                        st.error(f"Failed to update workout: {e}")
                                
                                if cancel_button:
                                    st.session_state.editing_workout_id = None
                                    st.rerun()
                        else:
                            st.error("Could not find the workout entry to edit.")
                            st.session_state.editing_workout_id = None # Clear if not found
                    else:
                        st.error("Failed to connect to database for editing.")
                        st.session_state.editing_workout_id = None
                
                # --- Delete Workout Confirmation ---
                if 'deleting_workout_id' in st.session_state and st.session_state.deleting_workout_id:
                    workout_id_to_delete_str = st.session_state.deleting_workout_id
                    st.warning(f"Are you sure you want to delete workout entry {workout_id_to_delete_str}? This action cannot be undone.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Confirm Delete", key=f"confirm_delete_{workout_id_to_delete_str}"):
                            try:
                                db_conn = get_db_connection()
                                if db_conn is not None:
                                    workouts_collection = get_workouts_collection(db_conn)
                                    query_id = ObjectId(workout_id_to_delete_str)
                                    
                                    result = workouts_collection.delete_one({"_id": query_id})
                                    
                                    if result.deleted_count > 0:
                                        st.success("Workout entry deleted successfully!")
                                    else:
                                        st.warning("Workout entry not found or already deleted.")
                                    st.session_state.deleting_workout_id = None
                                    st.rerun()
                                else:
                                    st.error("Failed to connect to the database for deletion.")
                            except Exception as e:
                                logger.error(f"Error deleting workout: {e}")
                                st.error(f"Failed to delete workout entry: {e}")
                                # Optionally keep deleting_workout_id to allow another attempt or require cancel
                                # st.session_state.deleting_workout_id = None # Or keep it to retry
                    with col2:
                        if st.button("Cancel Delete", key=f"cancel_delete_{workout_id_to_delete_str}"):
                            st.session_state.deleting_workout_id = None
                            st.rerun()


            with tab2:
                st.subheader("Weight Tracker")
                st.info(f"Displaying and logging entries for: {date_str}") # Added informational message
                # Visual separator was here, can be kept if desired after this markdown
                st.markdown("---") 
                
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
                df_all_weights = load_weight_data(username) # Load all weight data for the chart
                if not df_all_weights.empty:
                    st.line_chart(df_all_weights.set_index('Date')['Weight']) # Chart shows all data

                    # Filter entries for the selected date for listing, edit, and delete
                    df_selected_date_weights = df_all_weights[df_all_weights['Date'] == date_str]

                    st.subheader(f"Weight Entries for {date_str}") # Updated subheader
                    if not df_selected_date_weights.empty:
                        for index, entry in df_selected_date_weights.iterrows():
                            entry_id_str = str(entry['_id'])
                            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                            with col1:
                                st.write(f"Date: {entry['Date']}") # This will always be date_str
                            with col2:
                                st.write(f"Weight: {entry['Weight']} kg")
                            with col3:
                                if st.button("Edit", key=f"edit_weight_{entry_id_str}"):
                                    st.session_state.editing_weight_id = entry_id_str
                                    st.rerun()
                            with col4:
                                if st.button("Delete", key=f"delete_weight_{entry_id_str}"):
                                    st.session_state.deleting_weight_id = entry_id_str
                                    st.rerun()
                    else:
                        st.info(f"No weight entries recorded for {date_str}.")
                else:
                    st.info("No weight entries recorded yet for any date.") # Modified info message
                
                # Edit/delete forms will appear here based on session state (already implemented)
                if 'editing_weight_id' in st.session_state and st.session_state.editing_weight_id:
                    weight_to_edit_id_str = st.session_state.editing_weight_id
                    db_conn = get_db_connection()
                    if db_conn is not None:
                        # Ensure the weight entry being edited is for the currently selected date_str
                        # This check is important if the user changes selected_date while an edit form is open,
                        # though st.rerun() on date change should ideally prevent stale forms.
                        weight_to_edit = get_weight_by_id(db_conn, weight_to_edit_id_str)
                        
                        if weight_to_edit and weight_to_edit.get('Date') == date_str:
                            st.subheader(f"Edit Weight Entry for {weight_to_edit.get('Date', 'N/A')}")
                            with st.form(key="edit_weight_form"):
                                edited_weight_val = st.number_input(
                                    "Weight (kg):", 
                                    value=float(weight_to_edit['Weight']), 
                                    min_value=0.0, 
                                    step=0.1,
                                    key=f"edit_weight_val_{weight_to_edit_id_str}"
                                )

                                save_weight_changes = st.form_submit_button("Save Changes")
                                cancel_edit_weight = st.form_submit_button("Cancel")

                                if save_weight_changes:
                                    try:
                                        weights_collection = get_weights_collection(db_conn)
                                        query_id = ObjectId(weight_to_edit_id_str)
                                        
                                        update_data = {"$set": {"Weight": edited_weight_val, "Date": date_str}} # Ensure date is saved/updated
                                        result = weights_collection.update_one({"_id": query_id}, update_data)
                                        
                                        if result.modified_count > 0:
                                            st.success("Weight entry updated successfully!")
                                        else:
                                            st.warning("No changes made or weight entry not found.")
                                        st.session_state.editing_weight_id = None
                                        st.rerun()
                                    except Exception as e:
                                        logger.error(f"Error updating weight entry: {e}")
                                        st.error(f"Failed to update weight entry: {e}")
                                
                                if cancel_edit_weight:
                                    st.session_state.editing_weight_id = None
                                    st.rerun()
                        elif weight_to_edit: # Entry found but not for current date_str
                            st.warning(f"The weight entry you were editing was for {weight_to_edit.get('Date')}. Selected date is now {date_str}. Cancelling edit.")
                            st.session_state.editing_weight_id = None
                            st.rerun()
                        else: # Entry not found
                            st.error("Could not find the weight entry to edit.")
                            st.session_state.editing_weight_id = None 
                    else:
                        st.error("Failed to connect to database for editing weight.")
                        st.session_state.editing_weight_id = None

                # --- Delete Weight Confirmation ---
                if 'deleting_weight_id' in st.session_state and st.session_state.deleting_weight_id:
                    weight_id_to_delete = st.session_state.deleting_weight_id
                    # Optional: Fetch and display details of the weight entry being deleted for better UX
                    # weight_to_delete_details = get_weight_by_id(get_db_connection(), weight_id_to_delete)
                    # date_of_entry_to_delete = weight_to_delete_details.get('Date', 'this entry') if weight_to_delete_details else 'this entry'
                    # st.warning(f"Are you sure you want to delete the weight entry for {date_of_entry_to_delete}? This action cannot be undone.")

                    st.warning(f"Are you sure you want to delete this weight entry? This action cannot be undone.")
                    
                    col1_del, col2_del = st.columns(2) 
                    with col1_del:
                        if st.button("Confirm Delete Weight", key=f"confirm_delete_weight_{weight_id_to_delete}"):
                            try:
                                db_conn_del = get_db_connection() 
                                if db_conn_del is not None:
                                    weights_collection_del = get_weights_collection(db_conn_del) 
                                    query_id_del = ObjectId(weight_id_to_delete) 
                                    
                                    # Ensure deleting for the correct date if strictness is needed, though _id is unique
                                    # result_del = weights_collection_del.delete_one({"_id": query_id_del, "Date": date_str})
                                    result_del = weights_collection_del.delete_one({"_id": query_id_del}) 
                                    
                                    if result_del.deleted_count > 0:
                                        st.success("Weight entry deleted successfully!")
                                    else:
                                        st.warning("Weight entry not found or already deleted.")
                                    st.session_state.deleting_weight_id = None
                                    st.rerun()
                                else:
                                    st.error("Failed to connect to the database for deletion.")
                            except Exception as e_del: 
                                logger.error(f"Error deleting weight entry: {e_del}")
                                st.error(f"Failed to delete weight entry: {e_del}")
                                
                    with col2_del:
                        if st.button("Cancel Delete Weight", key=f"cancel_delete_weight_{weight_id_to_delete}"):
                            st.session_state.deleting_weight_id = None
                            st.rerun()
            
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
                            # Ensure 'Weight' and 'Reps' are numeric
                            workout_data['Weight'] = pd.to_numeric(workout_data['Weight'])
                            workout_data['Reps'] = pd.to_numeric(workout_data['Reps'])

                            # Sort by Weight and then Reps in descending order to easily pick the PR
                            # For each exercise, the first row after sorting by Weight then Reps (desc) will be the PR
                            pr_data_list = []
                            for exercise, group in workout_data.groupby('Exercise'):
                                # Find the max weight for the current exercise
                                max_weight = group['Weight'].max()
                                # Filter records with max weight
                                max_weight_sets = group[group['Weight'] == max_weight]
                                # Among those, find the record with max reps
                                pr_set = max_weight_sets.sort_values(by='Reps', ascending=False).iloc[0]
                                pr_data_list.append(pr_set)
                            
                            if pr_data_list:
                                pr_data = pd.DataFrame(pr_data_list)
                                # Select and rename columns for display
                                pr_data = pr_data[['Exercise', 'Weight', 'Reps', 'Date']]
                                pr_data.columns = ['Exercise', 'Max Weight (kg)', 'Max Reps at Max Weight', 'Date of PR']
                            else:
                                pr_data = pd.DataFrame(columns=['Exercise', 'Max Weight (kg)', 'Max Reps at Max Weight', 'Date of PR'])
                        else:
                            pr_data = pd.DataFrame(columns=['Exercise', 'Max Weight (kg)', 'Max Reps at Max Weight', 'Date of PR'])

                        st.dataframe(pr_data, hide_index=True)
                            
                        # Show PR history for selected exercise
                        st.subheader("PR Progress Chart")
                        selected_exercise = st.selectbox(
                            "Select Exercise",
                            options=workout_data['Exercise'].unique(),
                            key="pr_progress_chart_selectbox" # Added a key for uniqueness
                        )
                        
                        exercise_progress = workout_data[workout_data['Exercise'] == selected_exercise]
                        if not exercise_progress.empty:
                            exercise_prs = exercise_progress.groupby('Date')['Weight'].max()
                            st.line_chart(exercise_prs)
                        else:
                            st.info(f"No data available for {selected_exercise} to show PR progress.")
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
                                df_display = exercise_dist.reset_index()
                                # Assuming columns are 'Exercise' and 'count' after reset_index()
                                # If Series name is different, this might need adjustment based on actual column names
                                df_display.columns = ['Exercise', 'Times Performed'] 
                                st.dataframe(df_display, hide_index=True)
                            
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
        init_db()
        
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
