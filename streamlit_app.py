import streamlit as st
import pickle
import numpy as np


st.title("Iris ML app")

model_path = "model.pkl"
# Loading the model
model = pickle.load(open(model_path,"rb"))

def predict_flower(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm,model):
    #Prepare the query data as required to pass for the model
    query = np.array([SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]).reshape(1,-1)
    prediction = model.predict(query)
    return prediction.item()

sepal_length = st.number_input(label="sepal_length",value=None,format="%.2f")
sepal_width = st.number_input(label="sepal_width",value=None,format="%.2f")
petal_length = st.number_input(label="petal_length",value=None,format="%.2f")
petal_width = st.number_input(label="petal_width",value=None,format="%.2f")

if st.button("Predict"):
    output = predict_flower(sepal_length, sepal_width, petal_length, petal_width,model)
    st.write(f"Prediction: {output}")