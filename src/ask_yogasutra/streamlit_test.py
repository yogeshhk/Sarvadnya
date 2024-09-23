# import streamlit as st
# import json
#
#
# def main():
#     st.title("Streamlit File Upload Test")
#
#     # File uploader
#     uploaded_file = st.file_uploader("Choose a JSON file", type="json")
#
#     if uploaded_file is not None:
#         st.write("File uploaded successfully!")
#         st.write("File details:")
#         st.json({
#             "Filename": uploaded_file.name,
#             "FileType": uploaded_file.type,
#             "FileSize": uploaded_file.size
#         })
#
#         try:
#             # Try to read and parse the JSON content
#             data = json.load(uploaded_file)
#             st.write("JSON content:")
#             st.json(data)
#         except json.JSONDecodeError:
#             st.error("The uploaded file is not a valid JSON.")
#         except Exception as e:
#             st.error(f"An error occurred while processing the file: {str(e)}")
#     else:
#         st.write("No file uploaded yet.")
#
#     # Test session state
#     if 'upload_count' not in st.session_state:
#         st.session_state.upload_count = 0
#
#     st.write(f"Number of files uploaded in this session: {st.session_state.upload_count}")
#
#     if uploaded_file is not None:
#         st.session_state.upload_count += 1
#         st.experimental_rerun()
#
#
# if __name__ == "__main__":
#     main()

import streamlit as st

uploaded_file = st.file_uploader("Choose a file", type=["json", "txt", "csv"])

if uploaded_file is not None:
    try:
        content = uploaded_file.read()
        st.text("File content:")
        st.write(content)
    except Exception as e:
        st.error(f"Error: {e}")


