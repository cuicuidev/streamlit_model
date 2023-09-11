from routes import main
from routes import about
from routes import eda
from routes import workflow
from routes import model

routepaths = {'Inicio' : main.mainRoute, # Landing Page: explicamos el prjecto por encima y el stack utilizado
              'Datos' : eda.edaRoute, # Hablamos del dataset
              'Modelo' : model.modelRoute, # Permitimos al usuario interactuar con el modelo final
              'Resultados' : workflow.workflowRoute, # Hablamos del proceso de preprocesamiento de datos y entrenamiento del modelo, de la elección de stack y de la toma de decisiones, etc.
              'Acerca de' : about.aboutRoute,
              # Añadir mas rutas
              }