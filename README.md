
![](http://cdn.nextgov.com/media/img/upload/2017/04/14/041417cyberprotectionNG.jpg)


# Intelegent_Lock
The purpose of the project is to create lock mechanism with face recognition and liveness detection. Inspration comes from[ Dr. Andrew Ng's video](https://www.youtube.com/watch?v=wr4rx0Spihs)

## What it does?

the stucture of the program is; 

1. takes image and checks with pretrained neular network if what it sees, alive human or printout, img from phone etc. 
2. if it is more than %75 confidente that it is alive thing than it checks it if anyone from known people who has access 
3. if it is someone the system knows, than unlock the door :D 

## How to run:

Please install this modules:

```
pip3 install tflearn, face_recognition, opencv-python
python3 main_code.py
```

![](https://github.com/AhmetHamzaEmra/Intelegent_Lock/blob/master/a.gif)


## How it works:

The project has two main parts, 

1. Liveness detection
2. Face Recognation 

To secure the room-office-etc. you want to have this lock, You shouldn't allow anyone who is just holding someones picture which has access to that location. For this reason liveness detection is important. Although this implementation is not the best one and needs many improvements, the method is very simple. Take a frame, and check if it contains any printed picture-phone-tablet etc. I used conv net for this step and feed it with live and not-live(picutes which contains those things listed above). Note: If you want to improve this system you can implement 3d convolution which will be also capturing very important features (such as eye-lips movements etc) to determine if the image is live. 

For the second part, algorithm used in the project is from [Adam Geitgey](https://medium.com/@ageitgey?source=post_header_lockup). In nutshell:

1. Pretrained network predicts face landmarks from the frame
2. Compare the predicted vector with know face embedings 

Honestly, this is very oversplified 😄 but you can find detailed explanation in this [blog post](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)




## Finetuning note:

If you are not using any additional data, please make sure you dont have any files in the "img-live" and "img-not-live" directories! 
