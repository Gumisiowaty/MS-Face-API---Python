import os
import io
import json
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import requests
from PIL import Image, ImageDraw

credential = json.load(open('AzureCloudKeys.json'))
API_KEY = credential['API_KEY']
ENDPOINT = credential['ENDPOINT']
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))
image_url = 'https://media.cos.pl/image/30c/1500x700/30c7c4308074144a3ca395be9463fb4f.jpg'
image_name = os.path.basename(image_url)

response_detected_faces = face_client.face.detect_with_url(
    image_url,
    detection_model='detection_03',
    recognition_model='recognition_04'
    )

if not response_detected_faces:
    raise Exception('Nie wykryto twarzy')
    
response_image = requests.get(image_url)
img = Image.open(io.BytesIO(response_image.content))
draw = ImageDraw.Draw(img)  

for face in response_detected_faces:
    rect = face.face_rectangle
    
    left = rect.left
    top = rect.top
    right = rect.width + left
    bottom = rect.height + top
    
    draw.rectangle(((left, top), (right, bottom)), outline='green', width=3)
    

img.show()
    
print('Na zdjęciu wykryto ' + str(len(response_detected_faces)) + ' twarzy')