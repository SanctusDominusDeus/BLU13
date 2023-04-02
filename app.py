from utils import *
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

baseline_columns = [
"race",
"gender",
"age",
"weight",
"time_in_hospital",
"medical_specialty",
"has_prosthesis",
"num_procedures",
"number_outpatient",
"number_emergency",
"number_inpatient",
"number_diagnoses",
"blood_type",
"hemoglobin_level",
"blood_transfusion",
"diuretics", "insulin", "change", "diabetesMed", "readmitted"]

from sklearn.base import BaseEstimator, TransformerMixin
class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        Xdata = X.copy()
        cols = baseline_columns.copy()
        cols.remove("readmitted")
        return Xdata[cols]

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
        #del observation['admission_id']
        #test = list(observation.keys())[0]
    except:
        #changed
        error = 'No observation data.'
        return jsonify({"admission_id":_id, "error": error})
    
    
    #implement the mapping for the valid values
    
    valid_category_map = {
                "admission_id":int() ,
                "patient_id":int(),
                "race": ['White','Caucasian','European','AfricanAmerican','EURO','Afro American','?','African American','WHITE','Asian','Black','Hispanic','Other','Latino','AFRICANAMERICAN'],
                "gender":['Male','Female','Unknown/Invalid'],
                "age": ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)', None],                
                "weight": ['?', None,'[50-75)','[25-50)','[75-100)','[125-150)','[100-125)','[0-25)','[150-175)','[175-200)'],
                "admission_type_code": [ 3. , 6. , 2. , 1. , 5. , 8. ,None, 4. , 7.],
                "discharge_disposition_code": [ 1. ,25. ,6. ,11. ,2. ,14. ,3. ,5. ,None, 18. ,4. ,13. ,22. ,24. ,8. ,23. ,7. ,28. ,15. ,16. ,9.],
                "admission_source_code": [ 1 ,17 ,7 ,5 ,6 ,4 ,2 ,20 ,8 ,9 ,3 ,22],
                "time_in_hospital": [2 ,4 ,3 ,6 ,1 ,10 ,9 ,11 ,7 ,8 ,5 ,13 ,14 ,12],
                "payer_code":['?','UN','MC','SP','DM','HM','MD','BC','CM','CP','WC','OG','PO','MP','OT','CH','SI'],
                "medical_specialty":['?','Family/GeneralPractice','InternalMedicine','Surgery-Neuro','Orthopedics-Reconstructive','Pulmonology','Surgery-General','Hematology/Oncology','Gastroenterology','Oncology','Emergency/Trauma','Cardiology','Neurology','Orthopedics','Nephrology','Surgery-Cardiovascular/Thoracic','Urology','Surgery-Vascular','ObstetricsandGynecology','Radiologist','Pediatrics'
                                    ,'Surgery-Cardiovascular','DCPTEAM', 'Podiatry' ,'Psychiatry'
                                    ,'Endocrinology' ,'Psychology' ,'PhysicalMedicineandRehabilitation'
                                    ,'Surgery-Thoracic' ,'Endocrinology-Metabolism' ,'Pediatrics-Endocrinology'
                                    ,'Hematology' ,'Osteopath' ,'Pediatrics-Pulmonology' ,'Otolaryngology'
                                    ,'Obstetrics' ,'Resident' ,'Pediatrics-CriticalCare' ,'Gynecology'
                                    ,'SurgicalSpecialty', 'Radiology' ,'Surgery-Plastic' ,'Hospitalist'
                                    ,'Pathology', 'Surgery-Colon&Rectal' ,'InfectiousDiseases'
                                    ,'Pediatrics-Hematology-Oncology' ,'Surgery-Maxillofacial'
                                    ,'Psychiatry-Child/Adolescent', 'Anesthesiology-Pediatric' ,'Anesthesiology'
                                    ,'PhysicianNotFound' ,'Cardiology-Pediatric' ,'Ophthalmology' ,'Surgeon'
                                    ,'Psychiatry-Addictive' ,'Pediatrics-Neurology'
                                    ,'Obsterics&Gynecology-GynecologicOnco', 'Rheumatology'
                                    ,'AllergyandImmunology'],
                "has_prosthesis": [False,True],
                "complete_vaccination_status":['Complete','Incomplete','None'],
                "num_lab_procedures": [],
                "num_procedures": [1,0,4,3,2,6,5],
                "num_medications": range(1,71),
                "number_outpatient": [ 0 ,2  ,1 , 3 , 6 , 9 , 4 , 7 , 5 ,15 ,22 , 8 ,20 ,18 ,10 ,28 ,12 ,24 ,13 ,11 ,23 ,35 ,19, 25, 14],
                "number_emergency": range(0,23),
                "number_inpatient": range(0,19),
                "diag_1":[],
                "diag_2":[],
                "diag_3":[],
                "number_diagnoses":range(1,17),
                "blood_type": ['A-','O+','A+','B+','O-','AB-','AB+','B-'],
                "hemoglobin_level": range(9, 20),#float
                "blood_transfusion": [True, False],
                "max_glu_serum": ['NONE','>200','None','>300','NORM','Norm'],
                "A1Cresult":['None','>8','>7', 'Norm'],
                "diuretics": ['No', 'Yes'],
                "insulin": ['No', 'Yes'],
                "change":['No','Ch'],
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
        
        #observation_data=jsonify(observation),
        observation_data=observation,
        predicted_readmitted = prediction
    )
    try:
        print(observation)
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
        del ret_dict['id']
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
