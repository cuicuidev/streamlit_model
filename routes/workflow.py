import streamlit as st
import pandas as pd
import plotly.express as px
import labels

def get_confusion_matrix():
    df = pd.read_csv('confusion_matrix_data.csv')

    lbls = [val for val in labels.LABELS.values()][:-1]
    fig = px.imshow(df,
                    x= lbls,
                    y= lbls,
                    # color_continuous_scale='Viridis',  # Choose your desired color scale
                    labels=dict(x='Predicted Labels', y='True Labels', color='Confusion')
                    )

    # Customize the layout
    fig.update_layout(# title='Confusion Matrix Heatmap',
                      xaxis_showticklabels=False,
                      yaxis_showticklabels=False,
                      # coloraxis_showscale=False,
                      height = 700,
                      )
    return fig

def get_loss_chart():
    df = pd.read_csv('history.csv')
    fig = px.line(df[['Loss', 'Val_Loss']], labels = dict(x='Epoch', y='Loss', color=''))
    return fig

def get_acc_chart():
    df = pd.read_csv('history.csv')
    fig = px.line(df[['Accuracy', 'Val_Accuracy']], labels = dict(x='Epoch', y='Accuracy', color=''))
    return fig

def workflowRoute():
    st.title(':heavy_check_mark: Resultados y flujo de trabajo')
    st.write('---')

    st.write('### Métricas :straight_ruler:')

    st.write('Tras la evaluación del modelo utilizando el dataset de test, hemos obtenido una métrica en precisión de 0.33. Esto quiere decir que \
             el modelo clasifica correctamente la imágen un tercio de las ocasiones.')
    
    st.plotly_chart(get_confusion_matrix(), use_container_width=True, config={'displayModeBar': False})
    
    st.write('### Flujo de trabajo :gear:')
    st.write('')

    st.image('assets/workflow.png')


    st.write('### Evolución del entrenamiento :chart_with_upwards_trend:')

    st.write('Se puede ver en la siguiente gráfica como han ido evolucionando las funciones de pérdida en entrenamiento y validación. \
             Durante las primeras épocas, la pérdida en validación estaba bajando ligeramente hasta llegar a lo que se ve que es un \
             mínimo alrededor de las 120 épocas. A partir de ahí, la pérdida no ha parado de subir en ningún momento, llegando a sobrepasar el valor \
             inicial. Por el contrario, la pérdida en entrenamiento ha ido decreciendo de manera constante y, según parece, podría seguir decreciendo aun más.')

    st.plotly_chart(get_loss_chart(), use_container_width=True, config={'displayModeBar': False})

    st.write('La evolución de la precisión nos muestra una historia diferente. Podemos ver como la precisión en entrenamiento ha subido de manera \
             constante y podría seguir subiendo aun más, llegando a una medida de casi 0.7. Sin embargo, la métrica en validación se \
             estancó en 0.33 mucho antes de terminar el entrenamiento, alrededor de la época 400. A pesar de que la pérdida en validación \
             ha ido empeorando a partir de la época 120, la precisión del modelo seguía mejorando.')
    
    st.plotly_chart(get_acc_chart(), use_container_width=True, config={'displayModeBar': False})

    st.write('Todos estos resultados apuntan a un modelo sobre-entrenado. No obstante, esto es deliberado, puesto que son las métricas de validación las que \
             tomamos como atributo de mejora. Durante el entrenamiento, se han monitorizando lás métricas de validación en las diferentes versiones del modelo. \
             Se tomó una serie de decisiones al respecto y se modificó la arquitectura del modelo, el learning rate y otros parámetros, acorde a los resultados \
             obtenidos en iteraciones previas. El modelo aquí expuesto es el mejor de entre 6 modelos diferentes que hemos entrenado intentando introducir \
             una mejora en cada versión sucesiva. Finalmente, se validaron las métricas con un tercer dataset apartado especialmente para este propósito.')
    
    st.write('### Consideraciones :thinking_face:')

    st.write('Creemos que el tamaño del dataset era bastante reducido para enseñarle a una red neural a discriminar entre 39 diferentes categorías. \
             Además, es difícil en muchas ocasiones señalar unos atributos diferenciativos entre vehículos que tengan que ver a su vez con el fabricante. \
             Si bien es cierto que cada marca de coches tiene algo especial, algún guiño o estilo propio, como pueden ser los faros delanteros de un Porsche \
             o la forma algo rectangular de los Bentley, muchas veces no es así. Asimismo, atributos como la rectangularidad pueden ser característicos de \
             una marca, pero pueden aparecer de forma esporádica en otras, lo que da lugar a confusión. Sin duda, la única característica notable \
             que siempre nos va a decir quien es el fabricante de un vehículo es el logo de fabricante. No obstante, el modelo no puede fijarse siempre en un \
             atributo que ocupa una sección muy pequeña de la imágen y no siempre está siquiera presente.')
    st.write('Por otro lado, han habido decisiones que nos hemos visto forzados a tomar que creemos que han impactado negativamente en el producto final. Lo más \
             limitante para nosotros ha sido la falta de recursos computacionales. Hemos perdido mucho potencial al no poder agrandar el input del modelo a una \
             resolución mayor, por ejemplo a 720p. Tampoco hemos podido utilizar kernels de mayor tamaño a 3x3 en los bloques de convolución, algo que podría \
             haber permitido al modelo extraer features más complejas y detalladas. Por último, nos hemos limitado a utilizar un solo canal, es decir, \
             las imágenes que recibía el modelo estaban en blanco y negro. Una combinación entre utilizar imágenes de mayor resolución, aplicar \
             kernels mas grandes en las primeras fases convolutivas y utilizar canales RGB le habría permitido al modelo identificar nuevos rasgos que \
             pueden llegar a ser importantes. El color de un coche puede ser crucial para discriminar entre una marca y otra, y el modelo podría \
             aprender a ignorar las condiciones lumínicas que afectan al color basándose en el contexto de la imágen. Igualmente, una resolución mayor \
             y kernels más grandes podrían permitir al modelo identificar mejor atributos mas pequeños y complejos como lo son los logos de las marcas, \
             un elemento crucial del que hemos hablado antes.')