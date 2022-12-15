from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

# Model

class Input(BaseModel):
    sex: str
    pclass: int
    age: int

class Output(BaseModel):
    prediction: float

# Service

def load_model(filename='./model/dt.sav') -> DecisionTreeClassifier:
    return joblib.load(filename)


def parse_request(input: Input) -> pd.DataFrame:
    data = {'Age': [0.0], 'Pclass_1': [0], 'Pclass_2': [0], 'Pclass_3': [0], 'Sex_female': [0], 'Sex_male': [0]}
    data['Age'] = float(input.age)
    data['Pclass_'+str(input.pclass)] = 1
    data['Sex_'+input.sex] = 1
    return pd.DataFrame(data=data)


def predict_service(input: Input) -> Output:
    input_df = parse_request(input)
    return Output(prediction=model.predict(input_df)[0])

# Controller

app = FastAPI()
model = load_model()

@app.get('/ping')
async def ping_controller():
    return 'pong'


@app.post('/predict')
async def predict_controller(input: Input):
    return predict_service(input)