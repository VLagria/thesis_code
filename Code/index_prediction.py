import streamlit as st

def main():
    st.title("Multi-Page Vertical Navigation Bar")

    # Define the navigation options
    navigation = st.sidebar.radio("Navigation", ["Prediction", "Train Model"])

    # Display different content based on the selected navigation option
    if navigation == "Prediction":
        show_prediction()
    elif navigation == "Train Model":
        show_train_model()

def show_prediction():
    st.header("Prediction Page")

    # File uploader for image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.write("Uploaded image:")
        st.image(uploaded_image, caption=uploaded_image.name, use_column_width=True)
    

    uploaded_file = st.file_uploader("Upload Model", type=["b5", "txt", "xlsx"])

    if uploaded_file is not None:
        st.write(f"File uploaded: {uploaded_file.name}")


def show_train_model():
    st.header("Train Model Page")
    st.write("Welcome to the Train Model page!")
    # Add train model-related content here

if __name__ == "__main__":
    main()
