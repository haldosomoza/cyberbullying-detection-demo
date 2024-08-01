# importing Flask libraries
from flask import Flask, request
from flask_cors import CORS

# initializing Flask app
app = Flask(__name__)
CORS(app)

# defining the context for debugging
if __name__ == '__main__':
    app.run(debug=False)

# defining a function to print and log messages
def printAndLog(text):
    print(text)
    app.logger.info(text)

#=== === === === === === === 

# defining error handler for error 404
@app.errorhandler(404)
def error_handler_404(error):
    return "Page not found", 404    # Page Not Found

# defining error handler for any other
@app.errorhandler(Exception)
def error_handler_exception(error):
    return "An error occurred", 500 # Internal Server Error

#=== === === === === === ===

@app.route('/', methods=['GET'])
def api_get():
    # returning the first line of README.md file
    with open('README.md', 'r') as file:
        return file.readline()
    
#=== === === === === === ===

@app.route('/api/isAlive', methods=['GET'])
def api_isAlive_get():
    return { 'isAlive': True }

#=== === === === === === ===

# defining REST API endpoint for prediction
# resulting values between 0 (no bulling) to 1 (bullying)
@app.route('/api/predict', methods=['POST'])
def api_prediction_post():

    # getting the input parameters
    data     = request.get_json()
    userFrom = data.get('userFrom')
    userTo   = data.get('userTo')
    message  = data.get('message')

    # evaluating the message
    resultThisMsg = _evaluate_message(userFrom, message)
    # evaluating all messages from the user to the other user
    resultAllMsgs = _evaluate_all_messages(userFrom, userTo, message)
    
    # returning the results
    return { 
        'resultThisMsg': round(float(resultThisMsg),3), 
        'resultAllMsgs': round(float(resultAllMsgs),3) 
    }

#=== === === === === === ===

# evaluating the individual message
def _evaluate_message(userFrom, message):

    # logging the message
    printAndLog(f"Evaluating message  from user {userFrom}: {message}")

    # getting and returning the prediction
    return _get_prediction(message)

#=== === === === === === ===

all_messages = { }

# evaluating all messages from the user to the other user
def _evaluate_all_messages(userFrom, userTo, message):

    # getting the accumulated messages
    key_messages = f"{userFrom}||{userTo}"
    if key_messages not in all_messages:
        all_messages[key_messages] = { "messages": message + ". ", "prediction": None }
    else:
        all_messages[key_messages] = { "messages": all_messages[key_messages]["messages"] + message + ". ", "prediction": None }

    # logging the messages
    printAndLog(f"Evaluating messages from user {userFrom} to {userTo}: {all_messages[key_messages]['messages']}")
    
    # getting and keeping the prediction
    all_messages[key_messages]["prediction"] = _get_prediction(all_messages[key_messages]["messages"])
    # returning the prediction
    return all_messages[key_messages]["prediction"]

#=== === === === === ===
#=== === === === === ===
#=== === === === === ===

import json
import pickle
import pandas as pd

from app import printAndLog

from tensorflow.keras.models import load_model                      # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer           # type: ignore
from tensorflow.keras.preprocessing.text import tokenizer_from_json # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences   # type: ignore

#=== === === === === ===

class TheModels:

    # loading the tokenizer config and creating it
    tokenizer_config = json.load(open('saved_models/tokenizer_config.json'))
    tokenizer = tokenizer_from_json(tokenizer_config)
    printAndLog(f"Tokenizer loaded ...")

    # loading the model for scaling
    scaler = pickle.load(open('saved_models/scaler.pkl', 'rb'))
    printAndLog(f"Scaler loaded ...")

    # loading the model for prediction
    model = load_model('saved_models/model_03.h5')
    printAndLog(f"Model loaded ...")

    def __init__(self, value):
        self.instance_variable = value

#=== === === === === ===

def _get_prediction(message):
    
    # pre-processing the messages
    X_text, X_features = _preprocess(message)
    
    # executing the model
    prediction = _execute_model(X_text, X_features)

    # returning the prediction
    return prediction[0][0]

#=== === === === === ===

def _preprocess(message):

    # converting message to array
    message_array = [message]
    printAndLog(f"Message array: {message_array}")

    # tokenizing the message
    TheModels.tokenizer.fit_on_texts(message_array)
    printAndLog(f"Message fitted: {message_array}")
    X_text = TheModels.tokenizer.texts_to_sequences(message_array)
    printAndLog(f"Message tokenized: {X_text}")
    X_text = pad_sequences(X_text, maxlen=200)
    printAndLog(f"Message padded: {X_text}")

    #--- --- ---

    # getting the additional features
    df_feats    = pd.DataFrame({'sentiment_polarity': [0.3], 'sentiment_subjectivity': [0.5], 'dominant_topic': [0.7]})
    feats_array = df_feats[['sentiment_polarity', 'sentiment_subjectivity', 'dominant_topic']].values
    printAndLog(f"Features array: {feats_array}")

    #--- --- ---

    # normalizing/scaling the additional features
    X_features = TheModels.scaler.transform(feats_array)
    printAndLog(f"Features scaled: {X_features}")

    #--- --- ---

    # returning the pre-processed messages
    return X_text, X_features

#=== === === === === ===

def _execute_model(X_text, X_features):

    # executing the model and getting the prediction
    y_pred = TheModels.model.predict([X_text, X_features])
    printAndLog(f"Prediction: {y_pred}")

    # returning the prediction
    return y_pred

#=== === === === === ===