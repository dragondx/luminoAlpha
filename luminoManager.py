# handles application states and logics
# individual function contains every related to its own service
# common services are extracted for code reuse
import multiprocessing
import cv2
from dialogflow.dialogFlow import getDialogFlowResponse
from objRec.objectRec import ObjRec
from mqtt.mqtt import MQTT
from mqtt.pub import send as sendMQTTMessage

class Lumino:

    def __init__(self, verbose):
        self.verbose = verbose
        #  init all states here
        
        
        self.camQueue = multiprocessing.Queue(1)
        self.commandQueue = multiprocessing.Queue(2)


        self.startCameraFeed()
        self.objRecModule = ObjRec(self.camQueue)
        self.objRecModule.startYoloInference()
        
        self.network = MQTT(self.commandQueue)
        while True:
            self.main_logic_loop()

    def __del__(self):
        self.stopCameraFeed()


    # function for bluetooth class lifecycle


    # function for coral device


    # function for uploading to cloud services
 

    # functino for dialogflow




    ## CAMERA
    # function for camera
    # since code for camera is lightweight, could just do it here
    # handle the multiprocessing for camera, for different parts
    def startCameraFeed(self):  
        print("Initializing Camera Feed......")  
        self.camProc = multiprocessing.Process(target=self.camFeedLoop, args=(self.camQueue,))
        self.camProc.start()

    def stopCameraFeed(self):
        print("Stopping Camera Feed.......")
        self.camProc.terminate()

    def camFeedLoop(self, camQueue):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            camQueue.put(frame)


    # function for tf

    # main context loop



    # lifecycle loop for commanding individual function
    def main_logic_loop(self,):
        # Fetch new command
        if not self.commandQueue.empty():
            topic, payload = self.commandQueue.get()
            print((topic,payload))
            if topic == "DFReq":
                #DialogFlow Req handling
                print("Getting Response from Dialogflow")
                res = getDialogFlowResponse(payload, True)
                intent = res.intentName
                intentProb = res.intentConfidence
                fulfillment = res.fulfillmentText
                print(res)
                sendMQTTMessage("DFRes,"+fulfillment)
                # conditional selection here
                if intent == "ObjectRec":
                    topic = "ImageCaptionReq"
                elif intent == "FaceRec":
                    topic = "FaceReq"
                elif intent == "CharRec":
                    topic = "CharReq"
            
            
            if topic == "ImageCaptionReq":
                pass
            elif topic == "FaceReq":
                pass
            elif topic == "CharReq":
                pass
        
    # STOP lifecycles for 
    


    ## HELPER FUNCTIONS
    # function check status -- returns True if no problem found

lumino = Lumino(True)