from keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow.keras as keras

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Detección de impulsores de bomba detectuosos mediante Convolutional Neural Networks(CNN).
""")
st.write("### Autor: Amylkar Urrea Montoya. email: amylkar.urrea@udea.edu.co. [Linkedin](https://www.linkedin.com/in/amylkar-urrea-montoya-baab48196/)")
st.write("En trabajos que implican fundición es probable que los productos(en este caso impulsores de bomba) salgan defectuosos como consecuencia de un problema en el molde.")
st.write("En esta app se usa un modelo de clasificación construido con CNN para determinar si los impulsores de bomba son defectuosos o no.")
st.write('Lo único que debes hacer es subir una foto del impulsor de bomba haciendo click sobre el  botón **Browse files**')
st.write('En este [link](https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product) puedes descargar algunas fotos para hacer pruebas.')

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

images = False
if img_file_buffer:
    images = Image.open(img_file_buffer)

if images:
    model = load_model('best_model.h5')
    img = image.img_to_array(images)
    img = np.expand_dims(img, axis = 0)

    pred = model.predict(img)
    predictions = pred.tolist()[0]

    if predictions[0] > 0.5:
        prediction = "no defectuoso"
        probabilidad = predictions[0]
    else:
        prediction = "defectuoso"
        probabilidad = 1 - predictions[0]

    st.write("### La probabilidad de que el impulsor de bomba sea **{}** es del **{:.2f}%**".format(prediction, probabilidad*100))

col1, col2, col3 = st.beta_columns(3)
if images:
    col2.image(images, use_column_width=True)
