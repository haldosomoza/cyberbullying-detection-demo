# importing Flask libraries
from flask import Flask, request
from flask_cors import CORS

# importing the prediction module
import prediction 

# initializing Flask app
app = Flask(__name__)
CORS(app)

# defining the context for debugging
if __name__ == '__main__':
    app.run(debug=False)

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
    print(f"Evaluating message  from user {userFrom}: {message}")

    # getting and returning the prediction
    return prediction.get_prediction(message)

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
    print(f"Evaluating messages from user {userFrom} to {userTo}: {all_messages[key_messages]['messages']}")
    
    # getting and keeping the prediction
    all_messages[key_messages]["prediction"] = prediction.get_prediction(all_messages[key_messages]["messages"])
    # returning the prediction
    return all_messages[key_messages]["prediction"]

#=== === === === === === ===