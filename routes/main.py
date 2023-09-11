import streamlit as st

def mainRoute():
    
    st.title(':car: Clasificación de vehículos por fabricante')
    st.write(' _Aprovechando el poder de las redes neurales convolutivas para identificar el fabricante de un vehículo a partir de una imágen._')
    st.write(
        '---'
    )

    st.header('Problema y objetivo :mag:')
    st.write('Una de las mayores aplicaciones de tecnologías de Computer Vision está en los sistemas de vigilancia. La seguridad ciudadana es \
             una prioridad a lo largo de todo el mundo y tener acceso a unos sistemas avanzados ayuda a mejorarla. Cuando ocurre un crimen, las cámaras \
             de vigilancia son un vital aportador de evidencias.')
    st.write('Queremos facilitar este proceso haciendo uso de la inteligencia artificial. El objetivo de este proyecto es diseñar y entrenar un \
             modelo capaz de identificar el fabricante de un vehículo dada una imágen. El model será libre de ser usado tanto con imágenes cualesquiera, \
             como con frames individuales de una grabación de una cámara de seguridad.')

    col1, _, col2 = st.columns((3,1,8))

    col2.header('Herramientas :wrench:')
    col2.write('Para este proyecto teníamos la curiosidad de hacer uso, "pelearnos" y aprender más sobre una de las tantas herramientas que rodean el \
               Deep Learning como son las CNN (Convolutional Neural Network) o redes neurales convolutivas. Nos pareció interesante ver cómo podiamos \
               entrenar un modelo para que reconociese una parte de una imagen y fuese capaz de clasificarla. Para ello utilizamos Keras, un framework \
               muy popular para trabajar con redes neurales basado en Tensorflow, una biblioteca de herramientas que permite trabajar de manera eficiente \
               cont tensores aprovechando el hardware al máximo.')
    col1.write('')
    col1.write('')
    col1.write('')
    col1.write('')
    col1.image('assets/Tensorflow_logo.png', use_column_width = True)
    st.write('')
    st.write('')
    st.write('')

    col1, _, col2 = st.columns((8, 1, 3))

    col1.header('Datos :bar_chart:')
    col1.write('Ya que necesitabamos algo concreto como un dataset de imágenes decidimos buscar en Kaggle a ver que podíamos encontrar y después de charlarlo un poco entre todos nos decantamos por un dataset de imágenes que contenia entorno a unos 17.000 archivos de unas 39 marcas de coches diferentes, tenían buena calidad y se veían bien de tamaño para trabajar con ellas')
    col2.write('')
    col2.write('')
    col2.write('')
    col2.write('')
    col2.write('')
    col2.write('')
    col2.image('assets/Kaggle_logo.png', width = 180)