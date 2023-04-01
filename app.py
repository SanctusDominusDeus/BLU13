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
#DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')
DB = connect('sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

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
'''
@app.route('/predict', methods=['POST'])
def predict():

    obs_dict = request.get_json()

    try:
        _id = obs_dict['observation_id']
    except:
        error = 'Missing observation_id.'
        return jsonify({"observation_id":None,"error":error})
        #return {"observation_id":None,"error":error}
    
    try:
        observation = obs_dict['data']
    except:
        error = 'Missing data for the observation.'
        return jsonify({"observation_id":_id, "error": error})
        #return {"observation_id":_id, "error": error}
    
    
    #implement the mapping for the valid values
    
    valid_category_map = {
                "age": range(0,100),
                "sex": ['Male','Female'],
                "race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo','Other'],
                "workclass": ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov','Local-gov','?', 'Self-emp-inc', 'Without-pay', 'Never-worked'],
                "education": ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college','Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school','5th-6th', '10th', '1st-4th', 'Preschool', '12th'],
                "marital-status": ['Never-married', 'Married-civ-spouse', 'Divorced','Married-spouse-absent', 'Separated', 'Married-AF-spouse','Widowed'],
                "capital-gain": range(0,100000),
                "capital-loss": range(0,4357),
                "hours-per-week": range(0,24*7),
    }
    for key in observation.keys():
        if key not in valid_category_map.keys():
            error = '{} is not valid input.'.format(key)
            return jsonify({"observation_id":_id,"error":error})
            #return {"observation_id":_id,"error":error}
            
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return jsonify({"observation_id":_id,"error":error})
                #return {"observation_id":_id,"error":error}
        else:
            error = "Categorical field {} missing".format(key)
            return jsonify({"observation_id":_id,"error":error})
            #return {"observation_id":_id,"error":error}
    
    
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    prediction = pipeline.predict(obs)[0]
    proba = pipeline.predict_proba(obs)[0,1]

    response = dict()
    response['observation_id'] = _id
    response['prediction'] = prediction
    response['probability'] = proba
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)
    #return response

'''
@app.route('/predict', methods=['POST'])
def predict():
    # Flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json.
    obs_dict = request.get_json()
    _id = obs_dict['observation_id']
    observation = obs_dict['data']
    # Now do what we already learned in the notebooks about how to transform
    # a single observation into a dataframe that will work with a pipeline.
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    # Now get ourselves an actual prediction of the positive class.
    prediction = pipeline.predict(obs)[0]
    proba = pipeline.predict_proba(obs)[0, 1]
    ########response = {'observation_id':_id,'prediction': prediction,'probability': proba}
    response = {'prediction': prediction,'probability': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})
        #return {'error': error_msg}

@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([model_to_dict(obs) for obs in Prediction.select()])
    #return [model_to_dict(obs) for obs in Prediction.select()]

# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
