# Intelegent_Lock
The purpose of the project is to create lock mechanism with face recognition and liveness detection. Inspration comes from[ Dr. Andrew Ng's video](https://www.youtube.com/watch?v=wr4rx0Spihs)

## What it does?

the stucture of the program is; 

1. takes image and checks with pretrained neular network if what it sees, alive human or printout, img from phone etc. 
2. if it is more than %99.5 confidente that it is alive thing than it checks it if anyone from known people who has access 
3. if it is some it know, than unlock the door :D 

## How to run:

Please install this modules:

```
pip3 install tflearn, face_recognition, opencv-python
python3 main_code.py
```



## To do: 

* Imporve liveness detection ( I need bigger data set)
* Deploy it to IoT 