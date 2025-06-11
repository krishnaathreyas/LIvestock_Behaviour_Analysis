from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('livestock_behaviour.pkl')

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = []
    for question in [
        'milk_reduced', 'milk_abnormal', 'udder_swollen', 'udder_warm', 'avoids_milking',
        'limping', 'lies_down', 'stands_on_one_leg', 'leg_injury', 'reluctant_to_move',
        'aggression', 'nervous', 'stumbling', 'cannot_stand', 'teeth_grinding',
        'eating_normal', 'drinking_normal', 'social_behavior'
    ]:
        val = request.form.get(question)
        data.append(1 if val == 'Yes' else 0)

    # Temperature handling
    temp = request.form.get('temperature')
    if temp:
        try:
            temp = float(temp)
        except:
            temp = 0.0
    else:
        temp = 0.0
    data.append(temp)

    # Predict
    prediction = model.predict([data])[0]
    labels = {0: "Healthy", 1: "Lameness", 2: "Heat Stress"}
    result = labels.get(prediction, "Unknown")

    return render_template('form.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
