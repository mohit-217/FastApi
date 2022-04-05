import uvicorn
from fastapi import FastAPI
from hpsale import hpsale
import numpy as np
import pickle
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
app = FastAPI()
pickle_in = open("models/random_forest.pkl","rb")
f1=open("config/sku.json")
f2=open("config/city.json")
sku_dict=json.load(f1)
city_dict=json.load(f2)
classifier=pickle.load(pickle_in)
@app.post('/predict')
def predictsale(data:hpsale):
    data = data.dict()
    sku=data['sku']
    region=data['region']
    sku_count=sku_dict[sku]
    sku_region=city_dict[region]
    
    prediction = classifier.predict([[1, sku_region,2, sku_count,2019,2020,5,5,1,2020]])
    return {
        'prediction': int(prediction)
    }
