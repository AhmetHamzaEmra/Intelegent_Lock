import face_recognition
from os import listdir
from os.path import isfile, join
from glob import glob

def get_users():

    known_names=[]
    known_encods=[]

    for i in glob("people/*.jpg"):
        img = face_recognition.load_image_file(i)
        encoding = face_recognition.face_encodings(img)[0]
        known_encods.append(encoding)
        known_names.append(i[7:-4])

    return known_names, known_encods


