import pymongo

def get_db(connection_string):
    """
    Establishes a connection to MongoDB and returns the database object.

    Args:
        connection_string (str): The MongoDB connection string.

    Returns:
        pymongo.database.Database: The database object, or None if connection fails.
    """
    try:
        client = pymongo.MongoClient(connection_string)
        client.admin.command('ping')  # Verify connection
        db = client['GymTrackerDB']
        return db
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def get_users_collection(db):
    """
    Returns the 'users' collection from the database.

    Args:
        db (pymongo.database.Database): The database object.

    Returns:
        pymongo.collection.Collection: The 'users' collection object.
    """
    return db['users']

def get_workouts_collection(db):
    """
    Returns the 'workouts' collection from the database.

    Args:
        db (pymongo.database.Database): The database object.

    Returns:
        pymongo.collection.Collection: The 'workouts' collection object.
    """
    return db['workouts']

def get_weights_collection(db):
    """
    Returns the 'weights' collection from the database.

    Args:
        db (pymongo.database.Database): The database object.

    Returns:
        pymongo.collection.Collection: The 'weights' collection object.
    """
    return db['weights']

from bson import ObjectId

def get_workout_by_id(db, workout_id_str):
    """
    Fetches a single workout entry by its _id.

    Args:
        db (pymongo.database.Database): The database object.
        workout_id_str (str): The string representation of the workout's ObjectId.

    Returns:
        dict: The workout document if found, else None.
    """
    try:
        workouts_collection = get_workouts_collection(db)
        return workouts_collection.find_one({"_id": ObjectId(workout_id_str)})
    except Exception as e:
        # Consider logging the error e.g., using a logger object
        print(f"Error fetching workout by ID {workout_id_str}: {e}")
        return None

def get_weight_by_id(db, weight_id_str):
    """
    Fetches a single weight entry by its _id.

    Args:
        db (pymongo.database.Database): The database object.
        weight_id_str (str): The string representation of the weight's ObjectId.

    Returns:
        dict: The weight document if found, else None.
    """
    try:
        weights_collection = get_weights_collection(db)
        return weights_collection.find_one({"_id": ObjectId(weight_id_str)})
    except Exception as e:
        # Consider logging the error
        print(f"Error fetching weight by ID {weight_id_str}: {e}")
        return None

if __name__ == '__main__':
    # Example usage (optional, for testing)
    CONNECTION_STRING = "mongodb+srv://dev:dev@cluster0.hwutjuq.mongodb.net/"
    
    # Get the database object
    db_instance = get_db(CONNECTION_STRING)
    
    if db_instance:
        print(f"Successfully connected to database: {db_instance.name}")
        
        # Get collections
        users_collection = get_users_collection(db_instance)
        print(f"Users collection: {users_collection.name}")
        
        workouts_collection = get_workouts_collection(db_instance)
        print(f"Workouts collection: {workouts_collection.name}")
        
        weights_collection = get_weights_collection(db_instance)
        print(f"Weights collection: {weights_collection.name}")
        
        # Example: Insert a document into the users collection
        try:
            users_collection.insert_one({"name": "Test User", "email": "test@example.com"})
            print("Test user inserted successfully.")
        except Exception as e:
            print(f"Error inserting test user: {e}")
            
    else:
        print("Failed to connect to the database.")
