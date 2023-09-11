import streamlit as st
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, ReLU
from keras.optimizers import Adam
import keras.backend as K
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

from labels import LABELS

HEIGHT = 200
WIDTH = 200
N_CATEGORIES = 39

def load_model():
    
    # try:
    #     # Load the original model
    #     original_model = tf.keras.models.load_model('model.h5')

    #     original_model.save_weights('model_weights.h5')

    #     del original_model

    # except: pass

    model = Sequential()

    model.add(Input(shape=(HEIGHT, WIDTH, 1)))

    # First Conv Block
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Second Conv Block
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Third Conv Block
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Fourth Conv Block
    model.add(Conv2D(filters=256, kernel_size=3, padding='same', kernel_initializer='he_normal'))
    # model.add(BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(MaxPooling2D(pool_size=2))

    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(N_CATEGORIES, activation='softmax', kernel_initializer='he_normal'))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=10000,
                decay_rate=0.9)
    opt = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    model.load_weights('model_weights.h5')

    return model

def modelRoute():

    st.title(':eye: Modelo')
    
    st.write('Para ver el modelo en acción, suba una fotografía de un coche. Tenga en cuenta que el modelo está diseñado para categorizar \
             coches según su fabricante, por lo que si recibe una imágen que no es de un vehículo, intentará calificarla igualmente como si fuese uno. \
             Además, algunos fabricantes de vehículos no formaron parte del dataset, por lo que el modelo no los conoce y es incapaz de calificar \
             ninguna imágen con esa etiqueta. Estos son los fabricantes que puede clasificar:')
    
    with st.expander('Ver fabricantes'):
        st.write(list(LABELS.values()))
    
    model = load_model()

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Read the uploaded image using PIL
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Rescale the image to 200x200 pixels
        image = image.resize((200, 200))

        # Convert the PIL image to grayscale (single channel)
        image = image.convert('L')

        # Convert the grayscale image to a NumPy array
        image_array = tf.keras.preprocessing.image.img_to_array(image)

        # Create an ImageDataGenerator for normalization
        datagen = ImageDataGenerator(rescale=1.0/255.0)

        # Normalize the image
        normalized_image = datagen.standardize(image_array.reshape(1, 200, 200, 1))

        # Make a prediction using your pre-trained model
        prediction = model.predict(normalized_image)

        # Display the prediction
        st.write(f"### El fabricante del coche es _{LABELS[prediction.argmax()]}_.")