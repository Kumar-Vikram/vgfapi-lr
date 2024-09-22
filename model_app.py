import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

class Input(BaseModel):
    CONSOLE: object
    YEAR: int
    CATEGORY: object
    PUBLISHER: object
    RATING: object
    CRITICS_POINTS: float
    USER_POINTS:float

class Output(BaseModel):
    SalesInMillions: float
@app.post("/predict")
def predict(data: Input) -> Output:
    X_input = pd.DataFrame([[data.CONSOLE,data.YEAR,data.CATEGORY,data.PUBLISHER,data.RATING,data.CRITICS_POINTS,data.USER_POINTS]],columns=['CONSOLE', 'YEAR', 'CATEGORY', 'PUBLISHER',
       'RATING', 'CRITICS_POINTS', 'USER_POINTS'])
    # Load the model using pikle file
    model = joblib.load('video_game_sales_prediction.pkl')

    # predict using model
    prediction = model.predict(X_input)

    # return output
    return Output(SalesInMillions = prediction,msg="CICD Test")