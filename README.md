
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



## Finetuning note:

If you are not using any additional data, please make sure you dont have any files in the "img-live" and "img-not-live" directories! 