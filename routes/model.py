import streamlit as st
import tensorflow as tf
import numpy as np
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

        rescaled_image = rescale_img(image)
        # Convert the PIL image to grayscale (single channel)
        rescaled_image = rescaled_image.convert('L')
        
        first, second, third, fourth = get_feature_maps(model, rescaled_image)

        with st.expander('Visión del modelo'):
            st.write('## Imagen original')
            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.write(f'## Imagen {HEIGHT}x{WIDTH}px')
            st.image(rescaled_image, caption="Resized Image", use_column_width=True)

            if st.button('Generar feature maps'):
                st.write(f'### Primer bloque de convolución')
                st.image(next(first), caption="Feature Map", use_column_width=True)

                st.write(f'### Segundo bloque de convolución')
                st.image(next(second), caption="Feature Map", use_column_width=True)

                st.write(f'### Tercer bloque de convolución')
                st.image(next(third), caption="Feature Map", use_column_width=True)

                st.write(f'### Cuarto bloque de convolución')
                st.image(next(fourth), caption="Feature Map", use_column_width=True)

        # Convert the grayscale image to a NumPy array
        image_array = tf.keras.preprocessing.image.img_to_array(rescaled_image)

        # Create an ImageDataGenerator for normalization
        datagen = ImageDataGenerator(rescale=1.0/255.0)

        # Normalize the image
        normalized_image = datagen.standardize(image_array.reshape(1, HEIGHT, WIDTH, 1))

        # Make a prediction
        prediction = model.predict(normalized_image)

        # Display the prediction
        st.write(f"### El fabricante es _{LABELS[prediction.argmax()]}_.")

    else:
        # QUEREMOS NUESTRA SUPER ESTRELLITA!!!!!
        st.image('assets/miles_morales.png', use_column_width = True)
        st.write(f"### El fabricante es _{LABELS[40]}_.")

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

def rescale_img(img):
    return img.resize((HEIGHT, WIDTH))

def get_feature_maps(model, input_img):
    feature_maps_generators = []  # List to store generators for each convolutional block
    
    input_img_array = tf.keras.preprocessing.image.img_to_array(input_img)
    input_img_array = np.expand_dims(input_img_array, axis=0)
    
    # Create an ImageDataGenerator for normalization
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    normalized_input = datagen.standardize(input_img_array)

    # Preprocess the input image once for all layers
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=[layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)])
    intermediate_outputs = intermediate_model.predict(normalized_input)

    for intermediate_output in intermediate_outputs:
        def feature_map_generator(output):
            for i in range(output.shape[-1]):
                normalized_feature_map = (output[..., i:i+1] - np.min(output[..., i:i+1])) / (np.max(output[..., i:i+1]) - np.min(output[..., i:i+1]))
                yield normalized_feature_map

        feature_maps_generators.append(feature_map_generator(intermediate_output))
    
    return tuple(feature_maps_generators)


