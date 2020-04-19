import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import time
import multiprocessing

MQTT_SERVER = "localhost"
MQTT_PATH = "lumino_HW"


class MQTT:
    def __init__(self, commandQueue):
        # multiprocess start
        self.commandQueue = commandQueue
        self.networkProc = multiprocessing.Process(target=self.start_mqtt, args=(self.commandQueue,))
        self.networkProc.start()
        
    def __del__(self):
        self.networkProc.terminate()
        print("Ending network connection")

    def start_mqtt(self, commandQueue):
        self.commandQueue = commandQueue
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        self.client.connect(MQTT_SERVER, 1883, 60)
        
        # Blocking call that processes network traffic, dispatches callbacks and
        # handles reconnecting.
        # Other loop*() functions are available that give a threaded interface and a
        # manual interface.
        self.client.loop_forever()

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code "+str(rc))
    
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        self.client.subscribe(MQTT_PATH)
    
    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        print(msg.topic+" "+str(msg.payload))
        #display_text(str(msg.payload))
        if "IOSHelloGreetingsMessage" in str(msg.payload):
            time.sleep(2)
            publish.single(MQTT_PATH, "PIEcho", hostname=MQTT_SERVER)
            self.commandQueue.put(("tpoic","payload"))
            
    
        # more callbacks, etc


        

    
