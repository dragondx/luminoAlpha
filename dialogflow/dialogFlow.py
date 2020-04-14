import os
import dialogflow
from google.api_core.exceptions import InvalidArgument
from collections import namedtuple

DialogFlowQuery = namedtuple('DialogFlowQuery', 'queryText intentName intentConfidence  fulfillmentText')

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/pi/LuminoAlpha/dialogflow/luminoalpha.json'

DIALOGFLOW_PROJECT_ID = 'luminoalpha'
DIALOGFLOW_LANGUAGE_CODE = 'en'
SESSION_ID = 'me'

def getDialogFlowResponse(text, verbose):
    
    text_to_be_analyzed = text

    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
    text_input = dialogflow.types.TextInput(text=text_to_be_analyzed, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.types.QueryInput(text=text_input)
    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
    except InvalidArgument:
        raise

    if verbose:
        print("Query text:", response.query_result.query_text)
        print("Detected intent:", response.query_result.intent.display_name)
        print("Detected intent confidence:", response.query_result.intent_detection_confidence)
        print("Fulfillment text:", response.query_result.fulfillment_text)
    return DialogFlowQuery( response.query_result.query_text, response.query_result.intent.display_name, response.query_result.intent_detection_confidence, response.query_result.fulfillment_text)

# example:
# print(getDialogFlowResponse("hello", False).intentName)
