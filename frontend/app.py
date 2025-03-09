import os
import sys
import streamlit as st

# Add the current directory to the path so Python can find the frontend module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create __init__.py files if they don't exist to make the directories proper packages
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
frontend_utils_dir = os.path.join(frontend_dir, "utils")
frontend_data_dir = os.path.join(frontend_dir, "data")

# Ensure the frontend directory exists
if not os.path.exists(frontend_dir):
    os.makedirs(frontend_dir)
    print(f"Created directory: {frontend_dir}")

# Create frontend/__init__.py if it doesn't exist
frontend_init = os.path.join(frontend_dir, "__init__.py")
if not os.path.exists(frontend_init):
    with open(frontend_init, "w") as f:
        f.write("# Make frontend a package\n")
    print(f"Created file: {frontend_init}")

# Ensure the utils directory exists
if not os.path.exists(frontend_utils_dir):
    os.makedirs(frontend_utils_dir)
    print(f"Created directory: {frontend_utils_dir}")

# Create frontend/utils/__init__.py if it doesn't exist
utils_init = os.path.join(frontend_utils_dir, "__init__.py")
if not os.path.exists(utils_init):
    with open(utils_init, "w") as f:
        f.write("# Make utils a package\n")
    print(f"Created file: {utils_init}")

# Ensure the data directory exists
if not os.path.exists(frontend_data_dir):
    os.makedirs(frontend_data_dir)
    print(f"Created directory: {frontend_data_dir}")

# Create frontend/data/__init__.py if it doesn't exist
data_init = os.path.join(frontend_data_dir, "__init__.py")
if not os.path.exists(data_init):
    with open(data_init, "w") as f:
        f.write("# Make data a package\n")
    print(f"Created file: {data_init}")

# Now import from frontend.app
try:
    from frontend.app import main
    
    # Run the app
    if __name__ == "__main__":
        main()
except ImportError as e:
    st.error(f"Import error: {e}")
    st.write("Troubleshooting information:")
    st.write(f"Current directory: {os.getcwd()}")
    st.write(f"Python path: {sys.path}")
    st.write("Files in current directory:")
    for f in os.listdir("."):
        st.write(f"- {f}")
    if os.path.exists("frontend"):
        st.write("Files in frontend directory:")
        for f in os.listdir("frontend"):
            st.write(f"- {f}")
