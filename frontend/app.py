import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np

BASE_API_URL = "http://localhost:8000"  # FastAPI server URL

# Initialize session state variables if not already initialized
if 'true_verifications' not in st.session_state:
    st.session_state.true_verifications = []
if 'false_verifications' not in st.session_state:
    st.session_state.false_verifications = []

def get_registered_users():
    """Fetch registered users from the API."""
    response = requests.get(f"{BASE_API_URL}/list_users")
    if response.status_code == 200:
        return response.json().get("users", [])
    return []

def main():
    st.markdown(
        "<h1 style='background-color:#450b13; color:white; padding:10px; border-radius:10px; text-align:center;'>"
        "Face Registration & Verification</h1>",
        unsafe_allow_html=True
    )

    menu = ["Home", "Register", "Verify", "Delete User"]
    st.markdown(
        """
        <style>
            /* Change sidebar background color */
            [data-testid="stSidebar"] {
                background-color: #450b13 !important; /* Dark Red */
            }

            /* Change sidebar text color */
            [data-testid="stSidebar"] * {
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write('\n\n')
        st.write("Select an option from the sidebar")

        users = get_registered_users()

        if users:
            st.write("### 👥 Registered Users:")
            for user in users:
                st.write(f"- {user}")
        else:
            st.warning("No users registered yet.")

    elif choice == "Register":
        st.subheader("Register New User")
        name = st.text_input("Enter User Name")
        uploaded_files = st.file_uploader(
            "Upload multiple face images", 
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True
        )

        if st.button("Register"):
            if not name:
                st.error("Please enter a User ID")
                return

            if len(uploaded_files) < 3:
                st.error("Please upload at least 3 images")
                return

            files = [("files", file) for file in uploaded_files]
            response = requests.post(
                f"{BASE_API_URL}/register/{name}",
                files=files
            )

            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

    elif choice == "Verify":
        st.subheader("Verify User")
        
        # First, ask the user to select the real label
        uploaded_files = st.file_uploader(
            "Upload multiple face images for verification",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True
        )

        if uploaded_files:
            # Ask user to enter the real name of the person for verification
            real_label = st.selectbox(
                "Select the real label of the person",
                options=["None"] + get_registered_users()  
            )

            if real_label and st.button("Verify"):
                images = [("files", file) for file in uploaded_files]

                # Send images to backend for verification
                response = requests.post(
                    f"{BASE_API_URL}/verify",
                    files=images
                )

                if response.status_code == 200:
                    result = response.json()

                    # Track True/False based on the user's input
                    if result['verified']:
                        st.success(f"Person verified as {result['predicted_name']} (Confidence: {result['confidence']:.2%})")
                        if real_label == result['predicted_name']:
                            st.session_state.true_verifications.append(1)  
                        else:
                            st.session_state.false_verifications.append(1)  
                    else:
                        st.error(f"{result['predicted_name']} (Confidence: {result['confidence']:.2%})")
                        if real_label == "None":
                            st.session_state.true_verifications.append(1)  
                        else:
                            st.session_state.false_verifications.append(1)  

                    plot_verifications()

                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                

    elif choice == "Delete User":
        st.subheader("Delete User")

        # Get the list of registered users
        response = requests.get(f"{BASE_API_URL}/list_users") 
        if response.status_code == 200:
            registered_users = response.json()["users"]
        else:
            st.error("Failed to load registered users")
            return

        selected_user = st.selectbox("Select a user to delete", registered_users)

        if st.button("Delete User"):
            delete_response = requests.delete(f"{BASE_API_URL}/delete/{selected_user}")
            if delete_response.status_code == 200:
                st.success(f"User {selected_user} deleted successfully")
            else:
                st.error(delete_response.json().get("detail", "Failed to delete user"))

def plot_verifications():
    if st.session_state.true_verifications or st.session_state.false_verifications:
        # Count the true and false verifications
        true_count = sum(st.session_state.true_verifications)
        false_count = sum(st.session_state.false_verifications)

        labels = ['True verifications', 'False verifications']
        values = [true_count, false_count]

        fig, ax = plt.subplots(figsize=(6, 4)) 
        ax.bar(labels, values, color=['#450b13', '#450b13'])  
        ax.set_ylabel('Count')

        for i, value in enumerate(values):
            ax.text(i, value + 0.2, str(value), ha='center', va='bottom', fontsize=12)

        st.pyplot(fig)

if __name__ == "__main__":
    main()
