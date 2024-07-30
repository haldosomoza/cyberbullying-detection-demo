import json
import pickle
import pandas as pd

#from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import load_model                      # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer           # type: ignore
from tensorflow.keras.preprocessing.text import tokenizer_from_json # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences   # type: ignore

#=== === === === === ===

def get_prediction(message):
    
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
    print(f"Message array: {message_array}")

    # loading the tokenizer config and creating it
    tokenizer_config = json.load(open('saved_models/tokenizer_config.json'))
    tokenizer = tokenizer_from_json(tokenizer_config)
    print(f"Tokenizer loaded ...")
    
    # tokenizing the message
    tokenizer.fit_on_texts(message_array)
    print(f"Message fitted: {message_array}")
    X_text = tokenizer.texts_to_sequences(message_array)
    print(f"Message tokenized: {X_text}")
    X_text = pad_sequences(X_text, maxlen=200)
    print(f"Message padded: {X_text}")

    #--- --- ---

    # getting the additional features
    df_feats    = pd.DataFrame({'sentiment_polarity': [0.3], 'sentiment_subjectivity': [0.5], 'dominant_topic': [0.7]})
    feats_array = df_feats[['sentiment_polarity', 'sentiment_subjectivity', 'dominant_topic']].values
    print(f"Features array: {feats_array}")

    #--- --- ---

    # loading the model for scaling
    scaler = pickle.load(open('saved_models/scaler.pkl', 'rb'))
    print(f"Scaler loaded ...")

    # normalizing/scaling the additional features
    X_features = scaler.transform(feats_array)
    print(f"Features scaled: {X_features}")

    #--- --- ---

    # returning the pre-processed messages
    return X_text, X_features

#=== === === === === ===

def _execute_model(X_text, X_features):

    # loading the model for prediction
    model = load_model('saved_models/model_03.h5')
    print(f"Model loaded ...")

    # executing the model and getting the prediction
    y_pred = model.predict([X_text, X_features])
    print(f"Prediction: {y_pred}")

    # returning the prediction
    return y_pred

#=== === === === === ===