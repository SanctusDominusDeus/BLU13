import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
from uuid import uuid4


########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')
#DB = connect('sqlite:///predictions.db')

class Prediction(Model):
    #observation_id = IntegerField(unique=True)
    admission_id = IntegerField(unique=True)
    observation_data = TextField()
    #proba = FloatField()
    predicted_readmitted = TextField()
    actual_readmitted = TextField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    obs_dict = request.get_json()

    try:
        _id = obs_dict['admission_id']
    except:
        _id = None
        error = 'Missing admission_id.'
        return jsonify({"admission_id":_id,"error":error})
    
    try:
        #observation = obs_dict['data']
        observation = obs_dict
        del obs_dict['admission_id']
    except:
        #changed
        error = 'Unable to delete admission_id.'
        return jsonify({"admission_id":_id, "error": error})
    
    
    #implement the mapping for the valid values
    
    valid_category_map = {
                "patient_id":range(0,1000000),
                "race": [],
                "gender":['Male','Female'],
                "age": ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)', None],                
                "weight": [],
                "admission_type_code": range(1,9),
                "discharge_disposition_code": range(1,30),
                "admission_source_code": range(1,27),
                "time_in_hospital": range(0,15),
                "payer_code":[],
                "medical_specialty":[],
                "has_prosthesis": ['TRUE', 'FALSE'],
                "complete_vaccination_status":[],
                "num_lab_procedures": [],
                "num_procedures": range(1,10),
                "num_medications": range(1,100),
                "number_outpatient": range(1, 36),
                "number_emergency": range(1,23),
                "number_inpatient": range(1,19),
                "diag_1":[],
                "diag_2":[],
                "diag_3":[],
                "number_diagnoses":range(1,17),
                "blood_type": [],
                "hemoglobin_level": range(9, 20),
                "blood_transfusion": ['TRUE','FALSE'],
                "max_glu_serum": [],
                "A1Cresult":[],
                "diuretics": ['No', 'Yes'],
                "insulin": ['No', 'Yes'],
                "change":[],
                "diabetesMed":['No', 'Yes']
    }
    for key in observation.keys():
        if key not in valid_category_map.keys():
            error = '{} is not valid input.'.format(key)
            return jsonify({"observation_id":_id,"error":error})
            
    ''' 
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return jsonify({"observation_id":_id,"error":error})
        else:
            error = "Categorical field {} missing".format(key)
            return jsonify({"observation_id":_id,"error":error})
    '''
    
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    prediction = pipeline.predict(obs)[0]
    #proba = pipeline.predict_proba(obs)[0,1]

    response = dict()
    #response['admission_id'] = _id
    response['readmitted'] = prediction
    #response['prediction'] = bool(prediction)
    #response['probability'] = proba
    p = Prediction(
        admission_id=_id,
        #proba=proba,
        #observation=request.data
        observation_data=jsonify(observation),
        predicted_readmitted = prediction
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Admission ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.admission_id == obs['admission_id'])
        p.actual_readmitted = obs['readmitted']
        p.save()
        ret_dict = model_to_dict(p)
        del ret_dict['observation_data']
        return jsonify(ret_dict)
        #return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Admission ID: "{}" does not exist'.format(obs['admission_id'])
        return jsonify({'error': error_msg})

@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([model_to_dict(obs) for obs in Prediction.select()])

# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
