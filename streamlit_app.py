"""
Main entry point for Streamlit Cloud deployment
"""
import sys
import os
import streamlit as st

# Add the repository root to the path so Python can find the frontend module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create required __init__.py files to make frontend a proper package
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if not os.path.exists(frontend_dir):
    os.makedirs(frontend_dir)

utils_dir = os.path.join(frontend_dir, "utils")
if not os.path.exists(utils_dir):
    os.makedirs(utils_dir)

frontend_init = os.path.join(frontend_dir, "__init__.py")
if not os.path.exists(frontend_init):
    with open(frontend_init, "w") as f:
        f.write("# Make frontend a package\n")

utils_init = os.path.join(utils_dir, "__init__.py")
if not os.path.exists(utils_init):
    with open(utils_init, "w") as f:
        f.write("# Make utils a package\n")

# Now try to import main from frontend.app
try:
    # First try the direct import
    from frontend.app import main
except ImportError as e:
    # If that fails, provide helpful error information
    st.error(f"Import error: {e}")
    st.write("Troubleshooting information:")
    st.write(f"Current directory: {os.getcwd()}")
    st.write(f"Python path: {sys.path}")
    
    # Try an alternative approach - import the app directly and use its main function
    try:
        import frontend.app
        main = frontend.app.main
        st.success("Successfully imported main function using alternative method")
    except Exception as alt_e:
        st.error(f"Alternative import also failed: {alt_e}")
        st.write("Files in current directory:")
        for f in os.listdir("."):
            st.write(f"- {f}")
        
        if os.path.exists("frontend"):
            st.write("Files in frontend directory:")
            for f in os.listdir("frontend"):
                st.write(f"- {f}")
            
            if os.path.exists(os.path.join("frontend", "app.py")):
                st.write("Content of frontend/app.py:")
                with open(os.path.join("frontend", "app.py"), "r") as f:
                    st.code(f.read())
        
        # As a last resort, try running the app.py directly
        st.write("Attempting to run frontend/app.py directly as a fallback...")
        try:
            import runpy
            globals_dict = runpy.run_path(os.path.join("frontend", "app.py"))
            # Extract the main function from the globals
            if "main" in globals_dict:
                main = globals_dict["main"]
                st.success("Successfully extracted main function from app.py")
            else:
                available_functions = [name for name, obj in globals_dict.items() 
                                      if callable(obj) and not name.startswith("_")]
                st.write(f"Available functions in app.py: {available_functions}")
                st.error("Could not find 'main' function in app.py")
                sys.exit(1)
        except Exception as run_e:
            st.error(f"Failed to run app.py directly: {run_e}")
            sys.exit(1)

# Run the app
if __name__ == "__main__":
    main()
