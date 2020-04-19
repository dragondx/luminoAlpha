import paho.mqtt.publish as publish

MQTT_SERVER = "localhost"
MQTT_PATH = "lumino_HW"
 
def send(message):
    publish.single(MQTT_PATH, message, hostname=MQTT_SERVER)

send("test")