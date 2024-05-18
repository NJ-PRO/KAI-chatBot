import json
import re
import colorama
import pickle
import numpy as np
from tensorflow import keras
from flask import Flask, jsonify, request

colorama.init()

with open("./intents.json") as file:
    data = json.load(file)

def chatGen(inp):
    return "Hello! How can I help you today?"

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
    # res = chat(query)
    res = chatGen(query)
    return jsonify({"response" : res})
if __name__=="__main__":
    app.run(host="0.0.0.0",)