from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models




### Configuración de flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://starmonydev:Inf13rn0311530@localhost/starmonydb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '123447a47f563e90fe2db0f56b1b17be62378e31b7cfd3adc776c59ca4c75e2fc512c15f69bb38307d11d5d17a41a7936789'

api = Api(app)
db = SQLAlchemy(app)



### Modelos
class Tag(db.Model):
    __tablename__   = 'tag'
    
    id_tag      = db.Column(db.Integer,     primary_key=True, unique=True)
    name        = db.Column(db.String(45),  unique=True, nullable=False)
    description = db.Column(db.String(255), unique=True, nullable=False)
    value       = db.Column(db.Float,        nullable=False)

    tagsProgression = db.relationship('TagProgression', backref='tag')

    def __repr__ (self):
        return f'Tag({self.name},{self.value})'



class Progression(db.Model):
    __tablename__   = 'progression'

    id_progression  = db.Column(db.Integer,     primary_key=True, unique=True)
    name            = db.Column(db.String(50),  nullable=False)
    code            = db.Column(db.String(50),  unique=True, nullable=False)
    Symbol          = db.Column(db.String(50),  unique=True, nullable=False)

    tagsProgression     = db.relationship('TagProgression',         backref='progression')
    progressionGrades   = db.relationship('ProgressionGrade',       backref='progression')
    
    def __repr__ (self):
        return f'Progression({self.name},{self.symbol})'



class Scale(db.Model):
    __tablename__   = 'scale'

    id_scale= db.Column(db.Integer,     primary_key=True)
    name    = db.Column(db.String(45),  unique=True, nullable=False)
    symbol  = db.Column(db.String(20),  nullable=False)
    code    = db.Column(db.String(50),  unique=True, nullable=False)

    scaleGrades     = db.relationship('ScaleGrade',     backref='scale')
    concreteScales  = db.relationship('ConcreteScale',  backref='scale')

    def __repr__ (self):
        return f'Scale({self.name},{self.code})'



class Chord(db.Model):
    __tablename__   = 'chord'

    id_chord= db.Column(db.Integer,     primary_key=True)
    name    = db.Column(db.String(60),  unique=True, nullable=False)
    symbol  = db.Column(db.String(65),  unique=True, nullable=False)
    code    = db.Column(db.String(65),  unique=True, nullable=False)

    scaleGrades     = db.relationship('ScaleGrade', backref='chord')
 
    def __repr__ (self):
        return f'Chord({self.name},{self.symbol},{self.code})'



class Note(db.Model):
    __tablename__   = 'note'

    id_note = db.Column(db.Integer,     primary_key = True)
    name    = db.Column(db.String(45),  nullable=False, unique = True)
    symbol  = db.Column(db.String(45),  nullable=False, unique = True)

    concreteScales  = db.relationship('ConcreteScale',  backref='note')
    concreteChord   = db.relationship('ConcreteChord',  backref='note')

    def __repr__ (self):
        return f'Note({self.name},{self.symbol})'



class TagProgression(db.Model):
    __tablename__   = 'tag_progression'

    id_progression  = db.Column('progression_id_progression', db.Integer, db.ForeignKey('progression.id_progression'), primary_key = True)
    id_tag          = db.Column('tag_id_tag', db.Integer, db.ForeignKey('tag.id_tag'), primary_key = True)

    def __repr__ (self):
        return f'TagProgression({self.id_progression},{self.id_tag})'



class ConcreteChord(db.Model):
    __tablename__   = 'concrete_chord'

    id_concrete_chord   = db.Column('id_concrete_chord',    db.Integer, primary_key = True)
    id_note             = db.Column('note_id_note',         db.Integer, db.ForeignKey('note.id_note'))
    id_chord            = db.Column('chord_id_chord',       db.Integer, db.ForeignKey('chord.id_chord'))
    position_note_chord = db.Column('position_note_chord',  db.Integer, primary_key = True)

    concreteProgression   = db.relationship(
        'ConcreteProgression',
        primaryjoin = 'and_(ConcreteChord.id_concrete_chord == ConcreteProgression.id_concrete_chord, '
                            'ConcreteChord.position_note_chord == ConcreteProgression.position_note_chord)')

    def __repr__ (self):
        return f'ConcreteChord({self.id_concrete_progression},{self.position_note_chord})'



class ScaleGrade(db.Model):
    __tablename__   = 'scale_grade'

    id_scale_grade  = db.Column('id_scale_grade', db.Integer,     primary_key = True)
    id_chord        = db.Column('chord_id_chord', db.Integer,     db.ForeignKey('chord.id_chord'), nullable=False)
    id_scale        = db.Column('scale_id_scale', db.Integer,     db.ForeignKey('scale.id_scale'), nullable=False)
    grade           = db.Column('grade',          db.String(10),  primary_key = True)

    progressionGrades   = db.relationship(
        'ProgressionGrade',
        primaryjoin = 'and_(ScaleGrade.id_scale_grade == ProgressionGrade.id_scale_grade, '
                            'ScaleGrade.grade == ProgressionGrade.grade)')

    def __repr__ (self):
        return f'ScaleGrade({self.id_scale_grade},{self.grade})'



class ConcreteScale(db.Model):
    __tablename__   = 'concrete_scale'
    
    id_concrete_scale   = db.Column('id_concrete_scale',  db.Integer, primary_key = True)
    id_note             = db.Column('note_id_note',       db.Integer, db.ForeignKey('note.id_note'), nullable=False)
    id_scale            = db.Column('scale_id_scale',     db.Integer, db.ForeignKey('scale.id_scale'), nullable=False)
    position_note_scale = db.Column('position_note_scale',db.Integer, primary_key = True)

    def __repr__ (self):
        return f'ConcreteScale({self.id_concrete_scale},{self.position_note_scale})'

    concreteProgressions = db.relationship(
        'ConcreteProgression',
        primaryjoin = 'and_(ConcreteScale.id_concrete_scale == ConcreteProgression.id_concrete_scale, '
                            'ConcreteScale.position_note_scale == ConcreteProgression.position_note_scale)')


class ProgressionGrade(db.Model):
    __tablename__       = 'progression_grade'

    id_progression_grade= db.Column('id_progression_grade',   db.Integer,     primary_key = True)
    position_grade      = db.Column('position_grade',         db.Integer,     primary_key = True)
    id_progression      = db.Column('id_progression',         db.Integer,     db.ForeignKey('progression.id_progression'),    nullable=False)
    id_scale_grade      = db.Column('id_scale_grade',         db.Integer,     db.ForeignKey('scale_grade.id_scale_grade'),    nullable=False)
    grade               = db.Column('grade',                  db.String(10),  db.ForeignKey('scale_grade.grade'),             nullable=False)

    concreteProgressions = db.relationship(
        'ConcreteProgression',
        primaryjoin = 'and_(ProgressionGrade.id_progression_grade == ConcreteProgression.id_progression_grade, '
                            'ProgressionGrade.position_grade == ConcreteProgression.position_grade)')

    def __repr__ (self):
        return f'ProgressionGrade({self.id_progression_grade},{self.position_grade})'



class ConcreteProgression(db.Model):
    __tablename__       = 'concrete_progression'

    id_concrete_progression = db.Column('id_concrete_progression', db.Integer, primary_key=True, nullable=False)
    position_concrete_chord = db.Column('position_concrete_chord', db.Integer, primary_key=True, nullable=False)
    id_progression_grade    = db.Column('id_progression_grade', db.Integer, db.ForeignKey('progression_grade.id_progression_grade'), nullable=False)
    position_grade          = db.Column('position_grade',       db.Integer, db.ForeignKey('progression_grade.position_grade'),   nullable=False)
    id_concrete_chord       = db.Column('id_concrete_chord',    db.Integer, db.ForeignKey('concrete_chord.id_concrete_chord'),   nullable=False)
    position_note_chord     = db.Column('position_note_chord',  db.Integer, db.ForeignKey('concrete_chord.position_note_chord'), nullable=False)
    id_concrete_scale       = db.Column('id_concrete_scale',    db.Integer, db.ForeignKey('concrete_scale.id_concrete_scale'),   nullable=False)
    position_note_scale     = db.Column('position_note_scale',  db.Integer, db.ForeignKey('concrete_scale.position_note_scale'), nullable=False)

    def __repr__ (self):
        return f'ConcreteProgression({self.id_concrete_progression},{self.position_concrete_chord})'


def getDataset():
    #Entradas
    # Acorde
    # Escala
    # notas<>
    # Tag de Progression - Genero
    xConcreteChords = []
    xChords     = []
    xScales     = []
    xNotes      = []
    xTagsProg   = []

    bufferNotes = []
    bufferTags  = []

    maxTags  = 0

    #Salida
    # Acorde Concreto
    yConcreteChord  = []

    numberItems = ConcreteProgression.query.group_by("id_concrete_progression").count()

    for i in range(numberItems):
        if i == 0 :
            continue 
        concreteProgressions =   ConcreteProgression.query.filter_by(id_concrete_progression = i)
        concreteProgressionBuffer = []

        for concreteProgression in concreteProgressions:
            #Concrete Progression
            concreteProgressionBuffer.append( concreteProgression )

            #Scale
            progressionGrade = ProgressionGrade.query.filter_by(id_progression_grade = concreteProgression.id_progression_grade)[0]
            scaleGrade       = ScaleGrade.query.filter_by(id_scale_grade = progressionGrade.id_scale_grade)[0]
            xScales.append( scaleGrade.id_scale )

            concreteChords   =  ConcreteChord.query.filter_by(id_concrete_chord = concreteProgression.id_concrete_chord)
            
            #Notes
            notes = []
            for concreteChord in concreteChords:
                notes.append(concreteChord.id_note)
            
            noteArray = [0,0,0,0,0,0,0,0,0,0,0,0]
            for note in notes:
                noteArray[note - 1] = 1
            xNotes.append( noteArray )
       
            #Chord
            xChords.append(concreteChord.id_chord)

            #Tag
            tags = TagProgression.query.filter_by(id_progression = progressionGrade.id_progression)
            tagBuffer = []
            for tag in tags:
                tagBuffer.append(tag.id_tag)
            bufferTags.append( tagBuffer )
            if len(tagBuffer) > maxTags:
                maxTags = len(tagBuffer)

        lenCurrentResponse = len(concreteProgressionBuffer)
        for j in range(lenCurrentResponse):
            xConcreteChords.append( concreteProgressionBuffer[j].id_concrete_chord )
            if j == lenCurrentResponse - 1:
                yConcreteChord.append( concreteProgressionBuffer[0].id_concrete_chord )
            else: 
                yConcreteChord.append( concreteProgressionBuffer[ j + 1].id_concrete_chord )

    for tags in bufferTags:
        while len(tags) < maxTags:
            tags.append(0)
        xTagsProg.append( tags )

    return (xChords, xScales, xNotes, xTagsProg, xConcreteChords, yConcreteChord)


    

def convertToTensor(xChords, xScales, xNotes, xTagsProg, xConcreteChords, yConcreteChord, percent):
    numExamples         = len(xChords)
    numTestExamples     = round( numExamples * percent )
    numTrainExamples    = numExamples - numTestExamples

    xS = tf.constant( xScales, dtype=np.int32   )
    xT = tf.constant( xTagsProg, dtype=np.int32 )

    xCC = tf.one_hot( tf.constant(xConcreteChords, dtype=np.int32 ), 744 )

    xC = tf.one_hot( tf.constant(xChords, dtype=np.int32 ), 62)
    xN = tf.constant(xNotes, dtype=np.int32 )
    
    yC = tf.one_hot( tf.constant(yConcreteChord, dtype=np.int32 ), 744 )

    xSTrain = xS[0:numTrainExamples]
    xSTest  = xS[numTrainExamples:numExamples]

    xTDim   = len( xT[0] )
    xTTrain = tf.slice( xT, [0,0],                [numTrainExamples, xTDim] )
    xTTest  = tf.slice( xT, [numTrainExamples,0], [numTestExamples,  xTDim] )

    xCTrain = tf.slice( xC, [0,0],                [numTrainExamples, 62] )
    xCTest  = tf.slice( xC, [numTrainExamples,0], [numTestExamples,  62] )

    xNTrain = tf.slice( xN, [0,0],                [numTrainExamples, 12] )
    xNTest  = tf.slice( xN, [numTrainExamples,0], [numTestExamples,  12] )

    yCTrain = tf.slice( yC, [0,0],                [numTrainExamples, 744] )
    yCTest  = tf.slice( yC, [numTrainExamples,0], [numTestExamples,  744] )

    xCCTrain= tf.slice( xCC, [0,0],                [numTrainExamples, 744] )
    xCCTest  =tf.slice( xCC, [numTrainExamples,0], [numTestExamples,  744] )

    return (xSTrain, xSTest, xTTrain, xTTest, xCTrain, xCTest, xNTrain, xNTest, xCCTrain, xCCTest, yCTrain, yCTest)


def train():
    print("Obteniendo Datos...")
    [xChords, xScales, xNotes, xTagsProg, xConcreteChords, yConcreteChord] = getDataset()
    print("Datos obtenidos...")
    
    print("Convirtiendo Datos a Tensores...")
    [xSTrain, xSTest, xTTrain, xTTest, xCTrain, xCTest, xNTrain, xNTest, xCCTrain, xCCTest, yCTrain, yCTest] = convertToTensor( xChords, xScales, xNotes, xTagsProg, xConcreteChords,yConcreteChord, .1 )
    print("Tensores generados...")

    # Capas de Entrada
    chord_input = layers.Input(shape=(62,),  name='chords')
    notes_input = layers.Input(shape=(12,),  name='notes')
    scale_input = layers.Input(shape=(1,),   name='scales')
    tagsp_input = layers.Input(shape=(len(xTTrain[0]),),   name='tagsp')
    cchrd_input = layers.Input(shape=(744,), name='concrete_chord')


    # Capas de procesamiento de cada entrada
    chord_dense = layers.Dense(100, activation="relu")(chord_input)  #200 200 200 50 100 200
    notes_dense = layers.Dense(100, activation="relu")(notes_input) #200    - 3693
    scale_dense = layers.Dense(100, activation="relu")(scale_input)  #50     - 3442
    
    tagsp_dense = layers.Dense(50, activation="relu")(tagsp_input)  #50     - 3905
    cchrd_dense = layers.Dense(300, activation="relu")(cchrd_input)

    # Capa de concatenacion
    concat_cns_layer = layers.concatenate([ chord_dense, notes_dense, scale_dense])
    concat_cns_dense = layers.Dense(250, activation="sigmoid")(concat_cns_layer)

    concat_cct_layer = layers.concatenate([ cchrd_dense, tagsp_dense])
    concat_cct_dense = layers.Dense(250, activation="sigmoid")(concat_cct_layer)

    concat_out_layer = layers.concatenate([ concat_cns_dense, concat_cct_dense])

    # Capa de Salida
    concrete_chord_output = layers.Dense(744, activation="softmax", name='response')(concat_out_layer)

    # Definiendo el Modelo
    model = models.Model(
        inputs  = [ chord_input, notes_input, scale_input, cchrd_input, tagsp_input], 
        outputs = concrete_chord_output)

    # Compilcación del Modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"])

    # Resumen del Modelo
    model.summary()

    # Entrenamiento de la Red
    model.fit({'chords': xCTrain, 'notes': xNTrain, 'scales':xSTrain, 'concrete_chord': xCCTrain,'tagsp':xTTrain}, {'response': yCTrain}, epochs=200) 

    # Predicciones
    N_PREDS = 10                       
    preds = np.array( np.round( model.predict([xCTest[:N_PREDS], xNTest[:N_PREDS], xSTest[:N_PREDS], xCCTest[:N_PREDS],xTTest[:N_PREDS]]) ) )
    
    #Guardar el Modelo
    model.save('model.h5')


def plot():
    model = tf.keras.models.load_model('./../ia/model.h5')
    plot_model(model, to_file='model.png') #Exportar el diagrama del modelo del modelo a model.png en la ruta del programa


### API REST
@app.route('/api/chord/all')
def getAllChords():
    chords = Chord.query.all()
    for chord in chords:
        print(chord.symbol)
    return "hola";
    



@app.route('/api/progression/train')
def trainRNA():
    train()
    return "trined";



@app.route('/api/progression/concrete')
def getAllConcreteProgressions():
    model = tf.keras.models.load_model('./../ia/model.h5')
    return "hola";

