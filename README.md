# Overview
Off-line handwriting recognition involves the automatic conversion of text in an image into letter codes which are usable<br/> within computer and text-processing applications. Off-line handwriting recognition is comparatively difficult, as different<br/> people have different handwriting styles.<br/> 
In this problem statement we apply handwriting recognition on the images of both form and non forms. This application is very<br/> useful as there is a dire need to come up with technology to digitize the handwritten application, This topic is still a topic<br/> for active research and many researchers are working on the problem to come up with an application robust enough to handle all the different handwriting styles and interpret them. Major problems in case of forms is removing unwanted infromation and noise.

[This](https://docs.google.com/presentation/d/1niFtRGOegT9jFcqXAsJ1vefYUIYLhS0BVa5yx_mcdXM/edit?usp=sharing) is the link to the 
presentation explaining the algo in brief.
<br/>
Detailed description to be updated.

## Requirements<br/>
Python 3.5(only)<br/>
Tensorflow 1.4<br/>
OpenCV<br/>
Numpy

## Instructions to run
Get saved model from [here](https://drive.google.com/open?id=1fL94Hd4EE-tIfgYbrNmV22fkkd_r1SWP)
Put model in model folder inside hack then run the commands given below.
```
cd Hack
./compile
python3 MyModel.py
```
Output will be given in ans.txt file in format <br/>
x-coordinate y-coordinate width height recognized character.<br/>

coordinates are of oriented image and not the original one.<br/>

Test accuracy on NIST dataset is above 90%. 


