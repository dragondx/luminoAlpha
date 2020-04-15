# handles application states and logics
# individual function contains every related to its own service
# common services are extracted for code reuse
import multiprocessing
import cv2
from dialogflow.dialogFlow import getDialogFlowResponse
from objRec.objectRec import ObjRec

class Lumino:

    def __init__(self, verbose):
        self.verbose = verbose
        #  init all states here
        
        self.camQueue = multiprocessing.Queue(1)
        self.startCameraFeed()
        self.objRecModule = ObjRec(self.camQueue)
        self.objRecModule.startYoloInference()
        
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


    # START lifecycle for services


    # STOP lifecycles for services


    ## HELPER FUNCTIONS
    # function check status -- returns True if no problem found

lumino = Lumino(True)