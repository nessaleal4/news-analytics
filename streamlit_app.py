"""
Main entry point for Streamlit Cloud deployment
"""

# Import the main app from the frontend directory
import sys
import os

# Add the frontend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "frontend"))

# Import and run the main app
from frontend.app import main

# Run the app
if __name__ == "__main__":
    main()
