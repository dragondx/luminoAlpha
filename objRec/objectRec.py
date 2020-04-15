from objRec.inference import YOLO
from multiprocessing import Process
import time
class ObjRec:
    def __init__(self, camQueue):
        self.context = []
        self.camQueue = camQueue
        self.yolo = YOLO(self.camQueue)

# IMAGE CAPTIONING
    def getImageCaption(self):
        pass




# RT OBJREC
# Function to get the location of object
# MultiProcess implementation
    def startYoloInference(self):
        print("Starting Yolo")
        self.yolo.start_inference_async()
        
        
    def stopYoloInference(self):
        print("Terminating Yolo")
        self.yolo.terminate()

    def sampleOneSecOutput(self):
        #forConext
        print(self.yolo.toClassListVerbose())

    def isObjPresent(self, obj):
        return obj in self.yolo.toClassListVerbose()

