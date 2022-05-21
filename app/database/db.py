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
    position_of_chord         = []
    idConcreteScale           = []
    idConcreteChordCurrent    = []
    idConcreteChordResponse   = []
    idTags                    = {}
    numberItems = ConcreteProgression.query.group_by("id_concrete_progression").count()

    dataset = []

    for i in range(numberItems):
        if i == 0 :
            continue 
        currentResponse =   ConcreteProgression.query\
                            .filter_by(id_concrete_progression = i)\
                            .group_by("position_concrete_chord")
        
        bufferPosition= []
        bufferCurrent =  []
        bufferConcretescale = []
        bufferedProgressionGrade = []
        tags = []
        for concreteProgression in currentResponse:
            bufferPosition.append(concreteProgression.position_concrete_chord)
            bufferCurrent.append(concreteProgression.id_concrete_chord)
            bufferConcretescale.append( concreteProgression.id_concrete_scale )
            bufferedProgressionGrade    = ProgressionGrade.query.filter_by(id_progression_grade = concreteProgression.id_progression_grade)
            tags                        = TagProgression.query.filter_by(id_progression = bufferedProgressionGrade[0].id_progression)

        lenCurrentResponse = len(bufferCurrent)
        for j in range(lenCurrentResponse):
            position_of_chord.append(bufferPosition[j])
            idConcreteChordCurrent.append(bufferCurrent[j])
            idConcreteScale.append(bufferConcretescale[j])
            idTags[i] = tags

            if j == lenCurrentResponse - 1:
                idConcreteChordResponse.append( bufferCurrent[0] )
            else: 
                idConcreteChordResponse.append( bufferCurrent[j + 1] )
    
        for tag in tags:
            dataset.append([idConcreteScale[i - 1], position_of_chord[i - 1]])

    return (dataset, idConcreteChordCurrent, idConcreteChordResponse)


    

def convertToTensor(dataset, dataset2, response, percent):
    numExamples         = len(dataset)
    numTestExamples     = round( numExamples * percent )
    numTrainExamples    = numExamples - numTestExamples

    xDim = len( dataset[0] )

    xs  = tf.constant(dataset, dtype=np.int32 )
    xs2 = tf.one_hot( tf.constant( dataset2, dtype=np.int32), 744)
    ys  = tf.one_hot( tf.constant(response, dtype=np.int32), 744)
    
    xTrain  = tf.slice(xs, [0,0], [numTrainExamples, xDim])
    xTest   = tf.slice(xs, [numTrainExamples, 0], [numTestExamples, xDim])
    yTrain  = tf.slice(ys, [0,0], [numTrainExamples, 744])
    yTest   = tf.slice(ys, [numTrainExamples,0], [numTestExamples, 744])

    x2Train  = tf.slice(xs2, [0,0], [numTrainExamples, 744])
    x2Test   = tf.slice(xs2, [numTrainExamples,0], [numTestExamples, 744])

    return (xTrain, xTest, yTrain, yTest, x2Train, x2Test)




def train(nombre):
    print("Obteniendo Datos\n")
    [dataset, dataset2, response] = getDataset()
    print("Datos obtenidos\n")
    
    print("Convirtiendo Datos a Tensores\n")
    [xTrain, xTest, yTrain, yTest, x2Train, x2Test] = convertToTensor( dataset, dataset2,response, .2 )
    print("Tensores generados\n")

    capa_entrada1 = layers.Input(shape=(len( dataset[0],))  ) 
    dense_params = layers.Dense(300, activation="sigmoid")(capa_entrada1)
 
    capa_entrada2 = layers.Input(shape=(744,))
    dense_chord = layers.Dense(50, activation="sigmoid")(capa_entrada2)
 
    con = layers.concatenate([ dense_params, dense_chord])
    densa3 = layers.Dense(350, activation="sigmoid")(con)
    capa_salida = layers.Dense(744, activation="sigmoid", name='response')(densa3)

    modelo = models.Model(inputs=[capa_entrada1, capa_entrada2], outputs=capa_salida) 

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    modelo.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"])

    modelo.summary()
    modelo.fit({'input_1': xTrain, 'input_2': x2Train}, {'response': yTrain}, epochs=5000)

    N_PREDS = 10                       
    preds = np.array(np.round(modelo.predict([xTest[:N_PREDS], x2Test[:N_PREDS]])))
    print('np.argmax(a, axis=1): {0}'.format(np.argmax(preds, axis=1)))
    print('np.argmax(a, axis=1): {0}'.format(np.argmax( np.array(yTest[:N_PREDS]), axis=1)))

   
    #Guardar el Modelo
    modelo.save(nombre)
    return [modelo,preds]

def train_update():
    [model, preds] = train("update.h5")
    evaluacion = tf.estimator.evaluate(preds)
    print(evaluacion['accuracy'])

### API REST
@app.route('/api/chord/predict/<iC>/<iS>/<p>')
def getAllChords(iC, iS, p):
    model = tf.keras.models.load_model('model.h5')

    dataset = []
    position = []
    position.append( int(p) )
    dataset.append( [ int(iS), int(p) ] )
    xs  = tf.constant(dataset, dtype=np.int32 )
    xs2 = tf.one_hot( tf.constant( position , dtype=np.int32), 744)

    pred = model.predict( [xs, xs2] )
    resp = '{0}'.format(np.argmax(pred, axis=1))
    return resp;


@app.route('/api/progression/train')
def trainRNA():
    train("model.h5")
    return "trined";
