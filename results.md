## The results so far 

Baseline - Need to implement IoU score

Model -
* Flip=flip in car_dataset get_mask_and_regr
* Conv upsampling

Model architecture -
![](/images/kaggle_pku_architecture.jpg)

Explicit hyperparameters -
1. Batch Size
2. Num epochs 
    *  I think 200 to start with
    * Final model can have 2000 iterations
3. I think we should definetly decay the learning rate more as I think it is overshooting and exploding
4. Also ensemble models
5. Why not have two different models for the two different tasks
6. Use same xavier initialization
7. Dataset Augmentation

Tasks -
1. Determining model shapes is different from understanding it
2. Determine if flip should be present or not.

Notes -
1. Why are we predicting pitch as pitch sine and pitch cosine ?
2. What does function _regr_preprocess do ?

Quick questions ?

1. Is efficient net trained as well ?
Ans. Yes, but starting from saved weights

2. Why do we upsample ?
Ans. So that the bounding box locations can be determined. 

3. Is the model predicting object ID's?
Ans. No

4. How does the model work?
Ans. Takes input image, passes it through the model to get and 8,60,192 output. Each layer represents one feature predicted. For example mask is the centre of the vehicle. So the output map corresponding to it has the a shape pf 60,192 of 0s except for the vehicle centres which have a value of 1.

![mask](/images/mask.png)

5. Why are there 8 layers in the ouput conv map?
Ans. The pitch is pedicted as pitch cosine and pitch sine. This is addition to x,y,z,yaw and roll forms 7 values. Added to this is the mask creating 8 layers. Below are all the maps. (Note the color of the image has no meaning). (The images are all zeros except for the the pixel locations where mask value is 1, where the value is either +ve or -ve indicating x,y,z,pitch cosine,etc )

![regr](/images/regr.png)

6. Why not measure x,y values direcly from the mask map where ever pixel locations are 1 ?
Ans. Becasue the axis is relative to the camera location. Also we are predicting for a 3D world. So no 2D cordinate system on the mask map image will be aligned with it.

![Insert picture of car axis]()

7. Why reconstruct the entire output image ? For example if there are 12 cars in an image why not have a fully connected later with 12(or say 12x7 ( 7 because x,y,z,yaw,pitch cosine, pitch sine,roll) )output nodes directly predict value ?
Ans. Because we do not know the number of cars in the image before hand.



