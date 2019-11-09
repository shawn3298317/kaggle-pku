## The results so far 

Baseline - Need to implement IoU score
Model architecture -

Explicit hyperparameters -
1. Batch Size
2. Num epochs 
    *  I think 200 to start with
    * Final model can have 2000 iterations
3. I think we should definetly decay the learning rate more as I think it is overshooting and exploding
4. Also ensemble models
5. Why not have two different models for the two different tasks

Tasks -
1. Make sure that this model is not predicting the class
2. Determining model shapes is different from understanding it

Quick questions ?

1. Is efficient net trained as well ?
Ans. Yes, but starting from saved weights

2. Why do we upsample ?
Ans. So that the bounding box locations can be calculated