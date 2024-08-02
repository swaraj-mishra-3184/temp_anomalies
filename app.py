from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import OneHotEncoder
import google.generativeai as genai
import os
import html
import re

app = Flask(__name__)

# Load the trained model and encoder
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')

genai.configure(api_key="AIzaSyDI4oR8gtYuH1qJtgF26xq6p6pi8NJ6vqM") # Replace this with your gemini-api key 

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_input(avg_temp, year, month, country, encoder):
    country_encoded = encoder.transform([[country]])
    country_df = pd.DataFrame(country_encoded, columns=encoder.get_feature_names_out(['Country']))

    X_input = pd.DataFrame([[avg_temp, year, month]], columns=['AverageTemperature', 'year', 'month'])
    X_input = pd.concat([X_input, country_df], axis=1)

    return X_input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])
        month = int(request.form['month'])
        avg_temp = float(request.form['avg_temp'])
        country = request.form['country']

        input_data = preprocess_input(avg_temp, year, month, country, encoder)
        predicted_uncertainty = model.predict(input_data)[0]
        paragraph = generate_paragraph(avg_temp, predicted_uncertainty, year, month, country)
        return render_template('result.html', prediction=f"{predicted_uncertainty:.6f}", paragraph=paragraph)

    except Exception as e:
        return jsonify({'error': str(e)})

def generate_paragraph(avg_temp, predicted_uncertainty, year, month, country):
    month_name = pd.to_datetime(f"{year}-{month}-01").strftime("%B")

    prompt = (
        f"In {country}, the average temperature recorded for {month_name} {year} was {avg_temp:.2f}°C. "
        f"According to the model's predictions, the uncertainty in this temperature is estimated to be approximately {predicted_uncertainty:.2f}°C. "
        f"This level of uncertainty could significantly impact various sectors, including agriculture, energy consumption, and general climate conditions in {country}. "
        f"To mitigate these impacts, please provide preventive measures that can be taken in the following areas: "
        f"1. Agriculture: Consider strategies to adapt to changing temperatures and manage crop production. "
        f"2. Energy Consumption: Explore ways to enhance energy efficiency and manage energy use. "
        f"3. Climate Conditions: Identify steps to improve resilience to potential climate impacts. "
        f"Understanding and addressing these uncertainties is crucial for effective planning and decision-making in these areas."
    )

    generation_config = {
        "temperature": 0.45,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 1500,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    response = model.generate_content(prompt)
    return format_response(response.text)

def format_response(text):
    formatted_text = html.escape(text)
    formatted_text = formatted_text.strip()
    formatted_text = formatted_text.replace('\n', '<br>')
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_text)
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_text)

    return formatted_text

if __name__ == '__main__':
    app.run(debug=True)
