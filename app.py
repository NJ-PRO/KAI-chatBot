import json
import requests
import colorama
import pickle
import numpy as np
from tensorflow import keras
from flask import Flask, jsonify, request

colorama.init()

with open("./intents.json") as file:
    data = json.load(file)

def chatGen(inp):
    APP_ID = "256ee16d-06e0-4312-9b71-78ff9d472c1e"
    API_KEY = "sk46eff885448d6425b8ca1de173589e9b6023abcd3a395519a36a1cace5573f7e48005ff789bcf05ec2651c0f1033f60c6b166193bb8fa811b988deb816e4590e"

    response = requests.post("https://api.youai.ai/developer/v1/apps/run", 
                             json={
                                "appId": APP_ID,
                                "variables": {
                                    "demoVariable": inp
                                },
                                "workflow": "Main.flow"
                             },
                             headers={
                                 "Content-Type": "application/json",
                                 "authorization": f"Bearer {API_KEY}"
                             })
    
    if response.status_code == 200:
        data = response.json()
        return data.get('thread', 'Error: No thread found in response')
    else:
        return f"Error: Unable to reach API, status code: {response.status_code}"

def chat(inp):
    model = keras.models.load_model('chat_model')

    with open('./tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('./label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    max_len = 20

    while True:
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                return np.random.choice(i['responses'])

app = Flask(__name__)

@app.route("/bot", methods=["POST"])
def response():
    query = dict(request.form)['query']
    res = chat(query)
    # res = chatGen(query)
    return jsonify({"response" : res})
if __name__=="__main__":
    app.run(host="0.0.0.0",)