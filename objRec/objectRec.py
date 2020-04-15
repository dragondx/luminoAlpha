from inference import YOLO
from multiprocessing import Process
class ObjRec:
    def __init__(self):
        self.yolo = YOLO()

    def __del__(self):
        self.stopInference()

# IMAGE CAPTIONING
    def getImageCaption(self):
        pass




# RT OBJREC
# Function to get the location of object
# MultiProcess implementation
    def startInference(self):
        self.yolo.start_inference()
        print("go")
        
    def stopInference(self):
        self.yolo.terminate()

    def sampleOneSecOutput(self):
        self.yolo.toClassTable()
        print(self.yolo.toClassListVerbose())

    def isObjPresent(self, obj):
        return obj in self.yolo.toClassListVerbose()

obj = ObjRec()
obj.startInference()
while True:
    obj.sampleOneSecOutput()
