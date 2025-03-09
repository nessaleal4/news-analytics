"""
Main entry point for Streamlit Cloud deployment
"""
import sys
import os

# Add the repository root to the path so Python can find the frontend module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import from frontend.app
from frontend.app import main

# Run the app
if __name__ == "__main__":
    main()
