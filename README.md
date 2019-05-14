# Server-Carro
This is the main code of the server.

We have some files that are an essential part of the server.

  - requirements.txt are the Python libraries needed to run the server.
  - images/ is the folder where are going to be the images that the car send to the server.
  - inference_graph/ is the trained model and the label_map with the classes that the model can identify.
  - utils/ is a folder that contain all the tools that the model use.
  - server.py is the main code written in Flask that allows to send a base64 image and response with an action that the car is going to do
  
For the model we use Tensorflow Object Detection API https://github.com/tensorflow/models that give us a pre-trained model with the COCO dataset, we just take a infrastucture of a net and make transfer-learning with the weights of imagenet and make the net learn based on our dataset.
The model is Faster-RCNN-Inception-V2.
The dataset that we collected is here with some additional files of the training process: https://wetransfer.com/downloads/1ff3720731454ad709be5867d9e7c54020190512135203/9f0a38

Also we use the server file of the SunFounder_PiCar-V repo, we just eliminate the streaming module since we did some Python Scripts which takes photos and send them to the server, we did it because geting the image from the streaming is a quite slow. https://github.com/sunfounder/SunFounder_PiCar-V
