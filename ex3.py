import json
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw, ImageFont

credential = json.load(open('AzureCloudKeys.json'))
API_KEY = credential['API_KEY']
ENDPOINT = credential['ENDPOINT']
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))
image_file = open(r"E:\Studia\Cyfryzacja i zarządzanie danymi w biznesie\Rok II\Semestr II\Czerwonka\FaceAPI - Python\AzureVision\img\kubiak1.jpg", 'rb')

response_detection = face_client.face.detect_with_stream(
    image_file,
    detection_model='detection_01',
    recognition_model='recognition_04',
    return_face_attributes = ['age', 'emotion', 'gender']
    )

#print(vars(response_detection[0].face_attributes))

img = Image.open(image_file)
draw = ImageDraw.Draw(img)
myfont = ImageFont.truetype('C:\Windows\Fonts\Arial.ttf', 20)
color = (3, 206, 0 ) #kolor tekstu

for face in response_detection:
    age = face.face_attributes.age
    emotion = face.face_attributes.emotion
    gender = face.face_attributes.gender
    neutral = emotion.neutral * 100
    happiness = emotion.happiness * 100
    anger = emotion.anger * 100
    sadness = emotion.sadness * 100
    surprise = emotion.surprise * 100

    rect = face.face_rectangle
    left = rect.left
    top = rect.top
    right = rect.width + left
    bottom = rect.height + top

    draw.rectangle(((left, top), (right, bottom)), outline='green', width=3)
    draw.text((right + 4, top), 'Age: ' + str(int(age)), fill=(color), font = myfont)
    draw.text((right + 4, top+20), 'Gender: ' + gender, fill=(color), font = myfont)
    draw.text((right + 4, top+40), ' ')
    draw.text((right + 4, top+60), 'Emotions: ', fill=(color), font = myfont)
    draw.text((right + 4, top+80), 'Neutral: ' + str(neutral) + '%', fill=(color), font = myfont)
    draw.text((right + 4, top+100), 'Happiness: ' + str(happiness) + '%', fill=(color), font = myfont)
    draw.text((right + 4, top+120), 'Anger: ' + str(anger) + '%', fill=(color), font = myfont)
    draw.text((right + 4, top+140), 'Sadness: ' + str(sadness) + '%', fill=(color), font = myfont)
    draw.text((right + 4, top+160), 'Surprise: ' + str(surprise) + '%', fill=(color), font = myfont)

img.show()