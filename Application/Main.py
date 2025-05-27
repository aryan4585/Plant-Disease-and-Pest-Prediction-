import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load models
plant_model = tf.keras.models.load_model('D:\Plant Disease and Pest Prediction\Application\Plant_disease_trained_model.keras')
pest_model = tf.keras.models.load_model('D:\Plant Disease and Pest Prediction\Application\Pest_trained_model.keras')

# Load disease and pest info dictionaries from Python files
from disease_info import disease_info
from pest_info import pest_info

# Page configuration
st.set_page_config(page_title='Plant Disease and Pest Prediction', layout='wide')

# Navigation
menu = ['Home', 'About', 'Disease Recognition', 'Pest Recognition']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home':
    st.title('Plant Disease and Pest Prediction System')

    col1, col2 = st.columns(2)

    with col1:
        st.image('D:\Plant Disease and Pest Prediction\Application\home_image.jpg', use_column_width=True)

    with col2:
        st.image('D:\Plant Disease and Pest Prediction\Application\pest_home_image.jpg', use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease and Pest Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases and pests efficiently. Upload an image of a plant or a pest, and our system will analyze it to detect if any action should be taken or not. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** or **Pest Recognition** page and upload an image of a plant affected with disease or a pest.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential risks.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    - **All Information In One Place:** Receive description of the cause and also the action that should be taken.

    ### Get Started
    Click on the **Disease Recognition** or **Pest Recognition** page accordingly in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif choice == 'About':
    st.header('About the Project')
    st.markdown('This project uses convolutional neural networks to predict plant diseases and pests from images.')
    st.markdown("""
                #### About Plant Dataset 🌿
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 69K rgb images of healthy and diseased crop leaves which is categorized into 70 different classes.The total dataset is divided into 70/20/10 ratio of training, validation and test set preserving the directory structure.

                ##### Content
                1. train (53730 images)
                3. validation (15363 images)

                #### About Pest Dataset 🪱
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 13K rgb images of  Pests which is categorized into 36 different classes.The total dataset is divided into 70/20/10 ratio of training, validation and test set preserving the directory structure.

                ##### Content
                1. train (10210 images)
                3. validation (2950 images)
                """)

                

elif choice == 'Disease Recognition':
    st.header('Plant Disease Detection')
    uploaded_file = st.file_uploader('Upload a plant image', type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        image = image.resize((128, 128))
        input_arr = np.array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = plant_model.predict(input_arr)
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_name = list(disease_info.keys())[predicted_class]
        st.subheader(f'Prediction: {class_name}')
        st.write('**Description:**', disease_info[class_name]['description'])
        st.write('**Cure:**', disease_info[class_name]['cure'])

elif choice == 'Pest Recognition':
    st.header('Pest Recognition')
    uploaded_file = st.file_uploader('Upload a pest image', type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        image = image.resize((128, 128))
        input_arr = np.array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = pest_model.predict(input_arr)
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_name = list(pest_info.keys())[predicted_class]
        st.subheader(f'Prediction: {class_name}')
        st.write('**Description:**', pest_info[class_name]['description'])
        st.write('**Cure:**', pest_info[class_name]['cure'])
