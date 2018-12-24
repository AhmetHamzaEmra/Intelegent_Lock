
![](http://cdn.nextgov.com/media/img/upload/2017/04/14/041417cyberprotectionNG.jpg)


# Intelegent_Lock_2.0
As Andrej Karpathy said, Software2.0 is creating revalatrion in the field of software engineering. Now AI  is taking most of the heavy lifting in complexity and giving incredible results in the industry. One of this application is locks with face recognition. Although it sounds simple to combine these two technologies, it is more complex than most other applications. 



First of all, we have a problem with a dataset. Which is it is very hard to collect more than a couple of pictures of every person that we need to give access. If we can get a couple hundred pictures per person, One can easily train a simple model to do classification. But let us say we have 1000 employees that we want to give access. we need to get ~1.000.000 employee pictures and couple hundred pictures from other people in the World so the model can know if unknown people to lock to the door. On top of this, if we hire anybody else, We would need to train again. Screw that, we need another method. So we need a method that we can just use one picture per person to classify all of them and doesn't need to train when someone changes. That method is Landmark technique. Let's mark specific points in the face. Then we can use this transformation to classify people. Since everybody will have a different distance between two eyes or eyes and notice (except the twins!) we can tread this new measurement as embedding and if they are too close to each other in the multidimensional space we can they are the same person, vice versa. This is kind of an oversimplification, If you want to read more please check this [blog post](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) from Adam Geitgey.



Okay, since we solved the face recognition, we have another problem. What happens if someone holds a picture of me to get in. or How can we make sure they are live people and not someone with a mask. So we just need to classify if the input frames contain live people. For this stage, we can use a 3D convolutional network in our model. Why 3D you asked? Cause a frame does not contain some very useful features like eye movement, winkings etc. RNN also would work great but that part on you if you want to try out. (Please don't forget to send it to me 😄) 



Now, these two big problems should be solved. Please note that these two methods are not the best but they get the job done, but just keep in mind that I wouldn't use this project on my houses entrance 😅  Just use it in the meeting room or something 😄 



### TLDR:

- AI & ML good
- Landmark model for face recognition is crucial 
- Liveness is also crucial
- Not the safest lock tho (at least for now)



## What does it do?

The structure of the program is; 

1. takes image and checks with pretrained neular network if what it sees, alive human or printout, img from phone etc. 
2. if it is more than %95 confidente that it is an alive thing than it checks it if anyone from known people who have access 
3. if it is someone the system knows than unlock the door :D 



## How to run:

Please install this modules:

```
pip3 install keras, face_recognition, opencv-python
python3 main_code.py
```

![](https://github.com/AhmetHamzaEmra/Intelegent_Lock/blob/master/a.gif)



## References 

* [Software 2.0, Andrej Karpathy](https://medium.com/@karpathy/software-2-0-a64152b37c35)
* [Adam Geitgey](https://medium.com/@ageitgey)



**!WARNING!** 

Old version is store in v1 directory. It works faster but you wont get the same result. FYI!

**Important Not:**

Due to containging many of personal pictures of my friends, I cannot share the dataset!

