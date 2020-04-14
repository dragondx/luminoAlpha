import paho.mqtt.publish as publish
 
MQTT_SERVER = "192.168.86.72"
MQTT_PATH = "test_channel"
 
publish.single(MQTT_PATH, "This is a response from Raspberry Pi", hostname=MQTT_SERVER)