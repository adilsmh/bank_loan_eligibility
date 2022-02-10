from fastapi import FastAPI
import uvicorn

from model import loan_model, Client

app = FastAPI(title="Loan eligibility API",
              description="Submit client info to return eligibility with probability percentage",
              version="1.0.0")
model = loan_model()


@app.get("/predict")
def eligibility_prediction(client: Client):
    # define preprocessed data variable
    data = loan_model.preprocessing(client)
    # predict eligibility
    prediction, probability = model.predict_eligibility(data)

    # return prediction / probability
    return {"prediction": prediction[0],
            "probability": probability}


if __name__ == "__main__":
    uvicorn.run('app:app', host='0.0.0.0', port=5000)
