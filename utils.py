from sklearn.base import BaseEstimator, TransformerMixin

baseline_columns = [
# Maus
"race",
"gender",
"age",
"weight",
'admission_source_code',
'admission_type_code',
# Bom
"time_in_hospital",
"medical_specialty",
"has_prosthesis",
# Mau
"complete_vaccination_status",
# Bom
"num_procedures",
# Mau
#"num_medications",
'discharge_disposition_code',
# Bons
"number_outpatient",
"number_emergency",
"number_inpatient",
"number_diagnoses",
"blood_type",
"hemoglobin_level",
"blood_transfusion",
# Maus
"max_glu_serum",
"A1Cresult",
# Bons
"diuretics",
"insulin",
"change",
"diabetesMed",
"readmitted",
]



cat_cols = ['race','gender','age','weight','medical_specialty','complete_vaccination_status','blood_type','max_glu_serum',
'A1Cresult',
'diuretics',
'insulin',
'change',
'diabetesMed','admission_source_code',
'admission_type_code','discharge_disposition_code']

num_cols = [x for x in baseline_columns if x not in cat_cols and x != 'readmitted']

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
  
        return self

    def transform(self, X, y=None):
        Xdata = X.copy()
        cols = baseline_columns.copy()
        cols.remove("readmitted")

        return Xdata[cols]
    
class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
  
        return self

    def transform(self, X, y=None):
        Xdata = X.copy()
        Xdata['medical_specialty'] = Xdata['medical_specialty'].apply(clean_medical_specialty)
        #Xdata['weight'] = Xdata['weight'].apply(clean_weight)
        Xdata['age'] = Xdata['age'].apply(clean_age)
        Xdata['gender'] = Xdata['gender'].apply(clean_gender)
        Xdata['max_glu_serum'] = Xdata['max_glu_serum'].apply(clean_max_glu_serum)
        Xdata['A1Cresult'] = Xdata['A1Cresult'].apply(clean_a1cresult)
        Xdata['race'] = Xdata['race'].apply(clean_race)
        
        return Xdata
    



def clean_max_glu_serum(max_glu_serum):
    max_glu_serum = max_glu_serum.lower()
    if '200' in max_glu_serum:
        return '>200'
    elif '300' in max_glu_serum:
        return '>300'
    elif 'norm' in max_glu_serum:
        return 'norm'
    else:
        return 'none'

def clean_a1cresult(A1Cresult):
    if '>8' in A1Cresult:
        return '>8'
    elif '>7' in A1Cresult:
        return '>7'
    elif 'norm' in A1Cresult:
        return 'norm'
    else:
        return 'none'

def clean_race(race):
    race = str(race).lower()
    if ('afro' in race or 'african' in race) and 'american' in race:
        return 'africanamerican'
    elif 'euro' in race or 'caucasian'in race or 'white' in race:
        return 'white'
    elif 'asia' in race or 'yellow' in race:
        return 'asian'
    elif 'black' in race:
        return 'black'
    elif 'latin' in race or 'hispan' in race:
        return 'hispanic'
    else:
        return 'unknown'

def clean_gender(gender):
    gender = str(gender).lower()
    if 'female' in gender:
        return 'female'
    elif 'male' in gender:
        return 'male'
    else:
        return 'unknown'

def clean_age(age):
    age = str(age).lower()
    if '[0-10' in age:
        return '[0-10)'
    elif '10-20' in age:
        return '[10-20)'
    elif '20-30' in age:
        return '[20-30)'
    elif '30-40' in age:
        return '[30-40)'
    elif '40-50' in age:
        return '[40-50)'
    elif '50-60' in age:
        return '[50-60)'
    elif '60-70' in age:
        return '[60-70)'
    elif '70-80' in age:
        return '[70-80)'
    elif '80-90' in age:
        return '[80-90)'
    elif '90-100' in age:
        return '[90-100)'
    else:
        return 'unknown'




def clean_weight(weight):
    weight = str(weight).lower()
    if '[0-25' in weight:
        return '[0-25)'
    elif '25-50' in weight:
        return '[25-50)'
    elif '50-75' in weight:
        return '[50-75)'
    elif '75-100' in weight:
        return '[75-100)'
    elif '100-125' in weight:
        return '[100-125)'
    elif '125-150' in weight:
        return '[125-150)'
    elif '150-175' in weight:
        return '[150-175)'
    elif '175-200' in weight:
        return '[175-200)'
    else:
        return 'unknown'

def clean_medical_specialty(medical_specialty):
    categories = str(['Family/GeneralPractice', 'InternalMedicine', 'Surgery-Neuro', 'Orthopedics-Reconstructive', 'Pulmonology', 'Surgery-General', 'Hematology/Oncology', 'Gastroenterology', 'Oncology', 'Emergency/Trauma', 'Cardiology', 'Neurology', 'Orthopedics', 'Nephrology', 'Surgery-Cardiovascular/Thoracic', 'Urology', 'Surgery-Vascular', 'ObstetricsandGynecology', 'Radiologist', 'Pediatrics', 'Surgery-Cardiovascular', 'DCPTEAM', 'Podiatry', 'Psychiatry', 'Endocrinology', 'Psychology', 'PhysicalMedicineandRehabilitation', 'Surgery-Thoracic', 'Endocrinology-Metabolism', 'Pediatrics-Endocrinology', 'Hematology', 'Osteopath', 'Pediatrics-Pulmonology', 'Otolaryngology', 'Obstetrics', 'Resident', 'Pediatrics-CriticalCare', 'Gynecology', 'SurgicalSpecialty', 'Radiology', 'Surgery-Plastic', 'Hospitalist', 'Pathology', 'Surgery-Colon&Rectal', 'InfectiousDiseases', 'Pediatrics-Hematology-Oncology', 'Surgery-Maxillofacial', 'Psychiatry-Child/Adolescent', 'Anesthesiology-Pediatric', 'Anesthesiology', 'PhysicianNotFound', 'Cardiology-Pediatric', 'Ophthalmology', 'Surgeon', 'Psychiatry-Addictive', 'Pediatrics-Neurology', 'Obsterics&Gynecology-GynecologicOnco', 'Rheumatology', 'AllergyandImmunology']).lower()
    if str(medical_specialty).lower() in categories:
        return str(medical_specialty).lower()
    else:
        return 'unknown'