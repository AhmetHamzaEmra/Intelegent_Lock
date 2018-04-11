
import face_recognition
import cv2
import numpy as np 
font = cv2.FONT_HERSHEY_DUPLEX
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from sklearn.preprocessing import OneHotEncoder
from os import listdir
from os.path import isfile, join


live_img = ["liveness_detection/img-live/"+f for f in listdir("liveness_detection/img-live/") if isfile(join("liveness_detection/img-live/", f))]
live_label = [0 for i in range(len(live_img))]


not_live_img = ["liveness_detection/img-not-live/" + f for f in listdir("liveness_detection/img-not-live/") if isfile(join("liveness_detection/img-not-live/", f))]
not_live_label = [1 for i in range(len(not_live_img))]
print(live_img)

if live_img != [] and not_live_img != []:

    print("Liveness Model finetuning!")
    img = live_img + not_live_img
    labels = live_label+ not_live_label
    images=[]
    for i in img:
        img = cv2.imread(i, 0)
        img = cv2.resize(img, (100,100))
        images.append(img)

    X = np.array(images, dtype=float)
    y = np.array(labels, dtype=float)
    y= y.reshape((-1,1))
    X = X.reshape((-1,100,100,1))
    X /= 255
    Oneencoder = OneHotEncoder()
    y = Oneencoder.fit_transform(y)
    print("Data is ready!")
    print("Training is starting!")

    # Building convolutional network
    network = input_data(shape=[None, 100, 100, 1], name='input')
    network = conv_2d(network, 32, 5, activation='relu')
    network = avg_pool_2d(network, 2)
    network = conv_2d(network, 64, 5, activation='relu')
    network = avg_pool_2d(network, 2)
    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, 64, activation='relu')
    network = fully_connected(network, 2, activation='softmax',restore=False)
    network = regression(network, optimizer='adam', learning_rate=0.0001,
                         loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.load('model/my_model.tflearn')
    model.fit(X, y.toarray(), n_epoch=3, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=100,
          snapshot_epoch=False, run_id='model_finetuning')

    # # uncomment this part if you want to save finetuned model
    # model.save('model/my_model.tflearn')

    print("Finetuning is DONE!")
    print("Liveness Model is ready!")

else:
    
    # Building convolutional network
    network = input_data(shape=[None, 100, 100, 1], name='input')
    network = conv_2d(network, 32, 5, activation='relu')
    network = avg_pool_2d(network, 2)
    network = conv_2d(network, 64, 5, activation='relu')
    network = avg_pool_2d(network, 2)
    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, 64, activation='relu')
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.load('model/my_model.tflearn')
    print("Liveness Model is ready!")


  



video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
hamza_image = face_recognition.load_image_file("hamza.jpg")
hamza_face_encoding = face_recognition.face_encodings(hamza_image)[0]


known_names = ['HAMZA']
known_encods = [hamza_face_encoding]


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    liveimg = cv2.resize(frame, (100,100))
    liveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2GRAY)
    liveimg = np.array([liveimg/255])
    liveimg = liveimg.reshape((-1,100,100,1))
    pred = model.predict(liveimg)

    if pred[0][0]> .75:




        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            name = "Unknown"
            face_names = []
            for face_encoding in face_encodings:
                for ii in range(len(known_encods)):
                    # See if the face is a match for the known face(s)
                    match = face_recognition.compare_faces([known_encods[ii]], face_encoding)

                    

                    if match[0]:
                        name = known_names[ii]

                face_names.append(name)

        process_this_frame = not process_this_frame

        unlock = False
        for n in face_names:

            if n != 'Unknown':
                unlock=True

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            if unlock:
                cv2.putText(frame, 'UNLOCK', (frame.shape[1]//2, frame.shape[0]//2), font, 1.0, (255, 255, 255), 1)
            else:
                cv2.putText(frame, 'LOCKED!', (frame.shape[1]//2, frame.shape[0]//2), font, 1.0, (255, 255, 255), 1)
            


    else:
        cv2.putText(frame, 'WARNING!', (frame.shape[1]//2, frame.shape[0]//2), font, 1.0, (255, 255, 255), 1)
        # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()