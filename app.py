import os
import pickle
import traceback

import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

app = FastAPI()

PATH = "models"

@app.post("/continue-train")
def continue_train(model_name: str, train_input: UploadFile, new_model_name: str):
    try:
        model = model_from_file(model_name)

        train_data = pd.read_csv(train_input.file)

        # Map categorical variables
        train_data["sex"] = train_data["sex"].map({"Female": 0, "Male": 1})
        train_data["smoker"] = train_data["smoker"].map({"No": 0, "Yes": 1})
        train_data["day"] = train_data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
        train_data["time"] = train_data["time"].map({"Lunch": 0, "Dinner": 1})

        x = np.array(train_data[["total_bill", "sex", "smoker", "day", "time", "size"]])
        y = np.array(train_data["tip"])

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2
        }

        new_model_path = os.path.join(PATH, new_model_name + ".pickle")
        if os.path.exists(new_model_path):
            raise HTTPException(status_code=400, detail="New model name already exists. Choose a different name.")

        # Save the updated model
        with open(new_model_path, "wb") as new_model_file:
            pickle.dump(model, new_model_file)

        return {
            "metrics": metrics,
            "new_model_name": new_model_name + ".pickle"
        }

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/predict")
def predict(model_name: str, test_input: UploadFile):
    try:
        model = model_from_file(model_name)

        test_data = pd.read_csv(test_input.file)

        test_data["sex"] = test_data["sex"].map({"Female": 0, "Male": 1})
        test_data["smoker"] = test_data["smoker"].map({"No": 0, "Yes": 1})
        test_data["day"] = test_data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
        test_data["time"] = test_data["time"].map({"Lunch": 0, "Dinner": 1})

        x_test = test_data[["total_bill", "sex", "smoker", "day", "time", "size"]]

        predictions = model.predict(x_test)

        return {"Predictions": predictions.tolist()}

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)


@app.get("/models")
def models():
    return [os.path.splitext(f)[0] for f in os.listdir(PATH) if f.endswith('.pickle')]


def model_from_file(file_name):
    saved_model_name = file_name + ".pickle"
    model_path = os.path.join(PATH, saved_model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found.")

    with open(model_path, "rb") as saved_model_file:
        model = pickle.load(saved_model_file)
    return model
