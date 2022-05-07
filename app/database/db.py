from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
from flask_sqlalchemy import SQLAlchemy

import numpy as np                                                  #Librería de cálculo númerico para Python. Para instalar usar: `pip install numpy`
import pandas as pd                                                 #Para crear una tabla de datos. Para instalar Pandas usar 'pip install pandas'
import tensorflow as tf                                             #Para instalar Tensorlfow usar `pip install tensorflow`

from tensorflow.keras import layers, models                         #Los submodulos  para crear el modelo junto con las capas

from sklearn.datasets import load_breast_cancer                     #El conjunto de datos que vamos a usar. Para instalar Scikit-Learn usar `pip install scikit-learn`
from sklearn.model_selection import train_test_split                #Para dividir el conjunto de entrenamiento en train y test




### Configuración de flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://starmonydev:Inf13rn0311530@localhost/starmonydb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '123447a47f563e90fe2db0f56b1b17be62378e31b7cfd3adc776c59ca4c75e2fc512c15f69bb38307d11d5d17a41a7936789'

api = Api(app)
db = SQLAlchemy(app)



### Modelos
class Chord(db.Model):
    id_chord= db.Column(db.Integer,     primary_key=True)
    name    = db.Column(db.String(60),  unique=True, nullable=False)
    symbol  = db.Column(db.String(65),  unique=True, nullable=False)
    code    = db.Column(db.String(65),  unique=True, nullable=False)

    def __repr__(self):
        return '<Chord %r>' % self.symbol


class ConcreteProgression(db.Model):
    id_concrete_progression = db.Column(db.Integer,     primary_key=True, nullable=False)
    position_concrete_chord = db.Column(db.Integer,     primary_key=True, nullable=False)
    id_progression_grade    = db.Column(db.Integer,     nullable=False)
    position_grade          = db.Column(db.Integer,     nullable=False)
    id_concrete_chord       = db.Column(db.Integer,     nullable=False)
    position_note_chord     = db.Column(db.Integer,     nullable=False)
    
    def __repr__(self):
        return '<ConcreteProgression %r>' % self.id_concrete_scale_grade


### Entrenamiento
def train():
    # 1. Importación y procesado de datos
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    concreteProgressions = ConcreteProgression.query.all()
    
    dataConcreteProgresion = {
        "idConcreteProgression": {},
        "idProgressionGrade": {},
        "idConcreteChord": {},
        "positionConcreteChord": {},
        "positionGrade": {},
    }

    for it in concreteProgressions: {
        
    }

    
    X_train, X_test, y_train, y_test = train_test_split(X, y)


    df_train = pd.DataFrame(X_train, columns=columns)                  #No es necesario, tan solo por si quereís ver de forma visual los datos que tenemos

    # 2. Creación de la Red Neuronal
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

    capa_entrada = layers.Input(shape=(30,))                           #Capa de entrada, como cada dato tiene 30 variables (columnas), lo ponemos en la red neuronal
    densa_1 = layers.Dense(30, activation="relu")(capa_entrada)        #Capas densas, son las más simples que existen
    densa_2 = layers.Dense(20, activation="relu")(densa_1)             #Los parámetros son el número de unidades, y después la función de activación
    densa_3 = layers.Dense(10, activation="relu")(densa_2)             #Cada capa obtiene el input de la capa anterior
    capa_salida = layers.Dense(1, activation="sigmoid")(densa_3)       #última capa tenemos que poner las unidades y la función de activación según el número de clases

    modelo = models.Model(inputs=capa_entrada, outputs=capa_salida)    #Creamos el modelo indicandole la capa de entrada y de salida
    modelo.compile(optimizer="adam",                                   #Ponemos la función de optimización
                loss="binary_crossentropy",                         #Ponemos la función de perdidas
                metrics=["accuracy"])                               #La métrica de error que evaluará lo bien que lo hace nuestra red neuronal (la precisión)

    modelo.summary()                                                   #Imprime un resumen de nuestro modelo (no es necesario)

    # 3. Entrenamiento de la Red Neuronal
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    modelo.fit(X_train, y_train,                                       #Entrenamiento del modelo, pasandole las variables (betas) y el target a predecir (y)
            epochs=8)                                               #El número de epocas, una epoca es cuando el modelo en pasa por todos los datos y actualizar las betas

    # 4. Predicción de la red neuronal
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    N_PREDS = 3                                                        #Número de predicciones
    preds = np.round(modelo.predict(X_test[:N_PREDS]))                 #Hacemos tres predicciones y acertamos las tres.
    print(preds)
    print(y_test[:N_PREDS])





### API REST
@app.route('/api/chord/all')
def getAllChords():
    chords = Chord.query.all()
    for chord in chords:
        print(chord.symbol)
    return "hola";

@app.route('/api/progression/train')
def trainRNA():
    train();
    return "trined";


@app.route('/api/progression/concrete')
def getAllConcreteProgressions():
    concreteProgressions = ConcreteProgression.query.all()
    for it in concreteProgressions:
        print(it.id_concrete_progression)
    return "hola";

