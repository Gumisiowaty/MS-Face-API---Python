import json
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw

credential = json.load(open('AzureCloudKeys.json'))
API_KEY = credential['API_KEY']
ENDPOINT = credential['ENDPOINT']
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))
image_file = open(r"E:\Studia\Cyfryzacja i zarządzanie danymi w biznesie\Rok II\Semestr II\Czerwonka\FaceAPI - Python\AzureVision\img\team2.jpg", 'rb')

response_detected_faces = face_client.face.detect_with_stream(
    image = image_file,
    detection_model='detection_03',
    recognition_model='recognition_04',
    return_face_landmarks=True
    )

if not response_detected_faces:
    raise Exception('Nie wykryto twarzy')
    
img = Image.open(image_file)
draw = ImageDraw.Draw(img)

i = 1  
oznacz = 4 #ile twarzy zaznaczyć

for face in response_detected_faces:
    rect = face.face_rectangle
    
    #twarz
    left = rect.left
    top = rect.top
    right = rect.width + left
    bottom = rect.height + top
    
    draw.rectangle(((left, top), (right, bottom)), outline='green', width=3)
    
    #czubek nosa
    x = face.face_landmarks.nose_tip.x
    y = face.face_landmarks.nose_tip.y
    
    draw.ellipse(((x, y), (x+10, y+10)), fill='red')
    
    # usta
    mouth_left = face.face_landmarks.mouth_left.x, face.face_landmarks.mouth_left.y
    mouth_right = face.face_landmarks.mouth_right.x, face.face_landmarks.mouth_right.y
    lip_bottom = face.face_landmarks.under_lip_bottom.x, face.face_landmarks.under_lip_bottom.y
    
    draw.rectangle((mouth_left, (mouth_right[0], lip_bottom[1])), outline='yellow', width=2)
    
    #oczy
    left_eye = face.face_landmarks.eye_left_top.x-12, face.face_landmarks.eye_left_top.y-5
    right_eye = face.face_landmarks.eye_right_top.x-12, face.face_landmarks.eye_right_top.y-5
    
    draw.ellipse((left_eye, (left_eye[0]+25, left_eye[1]+25)), outline='blue', width=3)
    draw.ellipse((right_eye, (right_eye[0]+25, right_eye[1]+25)), outline='blue', width=3)
    
    
    if i == oznacz:
        break
    i+=1
    

img.show()
    

    
print('Na zdjęciu wykryto ' + str(len(response_detected_faces)) + ' twarzy')
print('Zaznaczono landmarki dla ' + str(i))