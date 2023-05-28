import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import numpy as np
from PIL import Image
import streamlit as st

features_list = pickle.load(open('images_features_list.pkl', 'rb'))
images_fils = pickle.load(open('images_fils.pkl', 'rb'))

model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

st.title('Fashion Recommdation System')
st.image('https://i.pinimg.com/originals/b4/6e/b7/b46eb746f7664083877a42aa05062dfe.jpg')

# save upload image

def save_upload_image(upload_image):
    try:
        with open(os.path.join('Upload', upload_image.name), 'wb') as i:
            i.write(upload_image.getbuffer())
        return 1
    except:
        return 0


# features extract
def extract_features(image_path, model):
    load_image = image.load_img(image_path, target_size = (224, 224))
    image_array = image.img_to_array(load_image)
    expanded_image_array = np.expand_dims(image_array, axis = 0)
    preprocess_image = preprocess_input(expanded_image_array)
    result_to_resnet = model.predict(preprocess_image)
    result = result_to_resnet.flatten()

    # normalizing
    normalized_result = result / norm(result)

    return normalized_result

# recommended
def image_recommended(img_features, img_feature_list):
    neighbors = NearestNeighbors(n_neighbors = 6, algorithm = 'brute', metric = 'euclidean')
    neighbors.fit(img_feature_list)

    distances, indices = neighbors.kneighbors([img_features])

    return indices

# file upload
st.success('Upload the Image')
upload_image = st.file_uploader('Choose an Image')
st.sidebar.header('All Departments')
st.balloons()
st.snow()


# Siderbar Menu
with st.sidebar:
    exp = st.expander("Men's Fashion", expanded = True)
    exp.text('T-shirts')
    exp.text('Shirts')
    exp.text('Jeans')
    exp.text('Cap')
    exp.text('shoes')
    exp.text('Bealt')
    exp.text('Wallet')

with st.sidebar:
    exp = st.expander("Girls's Fashion", expanded = True)
    exp.text('T-shirts')
    exp.text('Shirts')
    exp.text('Jeans')
    exp.text('Dress')
    exp.text('Scarf')
    exp.text('Ring')
    exp.text('Earrings')
    exp.text('Necklace')
    exp.text('Handbag')
    exp.text('Shoes')

with st.sidebar:
    exp = st.expander("Accessiories", expanded = True)
    exp.text('Watches')
    exp.text('Bags and Luggages')
    exp.text('Sunglasses')
    exp.text('Perfume')
    exp.text('Belt')

with st.sidebar:
    my_expander = st.expander("Stores", expanded=True)
    my_expander.text('Sportswear')
    my_expander.text('Gym clothes')
    my_expander.text('Swimsuit')
    my_expander.text('The Designer Boutique')
    my_expander.text('Fashion sales and deals')


if upload_image is not None:
    if save_upload_image(upload_image):
        # display the file
        display_image = Image.open(upload_image)
        st.image(display_image)

        # feature extract
        features = extract_features(os.path.join('Upload', upload_image.name), model)
        # st.text(features)

        # recommendention
        indices = image_recommended(features, features_list)

        # show
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(images_fils[indices[0][0]])

        with col2:
            st.image(images_fils[indices[0][1]])

        with col3:
            st.image(images_fils[indices[0][2]])

        with col4:
            st.image(images_fils[indices[0][3]])

        with col5:
            st.image(images_fils[indices[0][4]])
    else:
        st.error('Some error ocured in image upload')