from routes.main import mainRoute
from routes.about import aboutRoute
from routes.eda import edaRoute
from routes.workflow import workflowRoute
from routes.model import modelRoute

routepaths = {'Inicio' : mainRoute, # Landing Page: explicamos el prjecto por encima y el stack utilizado
              'Datos' : edaRoute, # Hablamos del dataset
              'Modelo' : modelRoute, # Permitimos al usuario interactuar con el modelo final
              'Resultados' : workflowRoute, # Hablamos del proceso de preprocesamiento de datos y entrenamiento del modelo, de la elección de stack y de la toma de decisiones, etc.
              'Acerca de' : aboutRoute,
              # Añadir mas rutas
              }