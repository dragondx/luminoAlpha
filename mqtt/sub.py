import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import time
 
MQTT_SERVER = "localhost"
MQTT_PATH = "test_channel"
 
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
 
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_PATH)
 
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
    #display_text(str(msg.payload))
    if "IOSHelloGreetingsMessage" in str(msg.payload):
        time.sleep(2)
        publish.single(MQTT_PATH, "PIEcho", hostname=MQTT_SERVER)
        
   
    # more callbacks, etc


	

    
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
 
client.connect(MQTT_SERVER, 1883, 60)
 
# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_forever()
