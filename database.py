from pymongo import MongoClient

# Connect to MongoDB (like opening a notebook)
client = MongoClient("mongodb://localhost:27017/")  # Default MongoDB URL
db = client["ai_visualizations"]  # Database name
collection = db["prompts"]  # Collection (like a folder)

def save_prompt_and_code(user_prompt, generated_code):
    """Saves the user's prompt and AI-generated code to MongoDB."""
    data = {
        "user_prompt": user_prompt,
        "generated_code": generated_code,
        "timestamp": datetime.now()  # Records when it was saved
    }
    collection.insert_one(data)  # Saves to database
    print("✅ Saved to MongoDB!")

def get_history():
    """Returns all past prompts + code from MongoDB."""
    return list(collection.find())  # Gets all saved records