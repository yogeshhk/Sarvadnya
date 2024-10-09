import streamlit as st

st.set_page_config(page_title="Yoga Sutras Explorer - Main Page", layout="wide")

def main():
    st.title("Yoga Sutras Explorer - Main Page")
    st.write("""
    Welcome to the Yoga Sutras Explorer! This application allows you to:
    
    1. Explore the connections between different sutras
    2. View and edit sutra details
    3. Understand the structure of the yoga sutras
    
    Navigate to the 'Graph Viewer' page in the sidebar to start exploring!
    """)

    # You might want to add some example content or images here

if __name__ == "__main__":
    main()