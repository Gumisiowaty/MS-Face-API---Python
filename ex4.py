import json
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw, ImageFont

credential = json.load(open('AzureCloudKeys.json'))
API_KEY = credential['API_KEY']
ENDPOINT = credential['ENDPOINT']
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))
#gdzie szukać twarzy
image_file = open(r"E:\Studia\Cyfryzacja i zarządzanie danymi w biznesie\Rok II\Semestr II\Czerwonka\FaceAPI - Python\AzureVision\img\kubiak-group.jpg", 'rb')

response_detected_faces = face_client.face.detect_with_stream(
    image = image_file,
    detection_model='detection_03',
    recognition_model='recognition_04'
    )

myface_ids = [face.face_id for face in response_detected_faces]

#twarz do znalezienia
szukana_twarz = open(r"E:\Studia\Cyfryzacja i zarządzanie danymi w biznesie\Rok II\Semestr II\Czerwonka\FaceAPI - Python\AzureVision\img\kubiak2.jpg", 'rb') 
response_face_kubiak = face_client.face.detect_with_stream(
    image = szukana_twarz,
    detection_model='detection_03',
    recognition_model='recognition_04'
    )

face_id_kubiak = response_face_kubiak[0].face_id


matched_faces = face_client.face.find_similar(
    face_id = face_id_kubiak,
    face_ids = myface_ids
    )


img = Image.open(image_file)
draw = ImageDraw.Draw(img)
myfont = ImageFont.truetype('C:\Windows\Fonts\Arial.ttf', 20)
color = (3, 206, 0 ) #kolor tekstu

for matched_face in matched_faces:
    for face in response_detected_faces:
        if face.face_id == matched_face.face_id:
            rect = face.face_rectangle
            left = rect.left
            top = rect.top
            right = rect.width + left
            bottom = rect.height + top
            
            pewnosc = matched_face.confidence * 100
            print('Pewnosc: ' + str(round(pewnosc, 2)) + '%')
            
            draw.text((right + 4, top), 'Pewnosc: ' + str(round(pewnosc, 2)) + '%', fill=(color), font = myfont)
            draw.rectangle(((left, top), (right, bottom)), outline='green', width=3)

img.show()