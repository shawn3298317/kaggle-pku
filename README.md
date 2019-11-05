#### kaggle-pku
Team repo. for PKU Autnomous Driving Kaggle Competition.

To run the code -
1. If you are training locally ```Do pyhton train_model.py ```
2. If you are using the SSC cluster do - ```qsub scripts/qsub_job.template.sh```

Code explained -
1. This code uses [CenterNet](https://arxiv.org/pdf/1904.07850.pdf) to draw 3D bounding boxes for objects -

![task](images/task.JPG)

The quantities to be estimated are the -

* X, Y, Z co-ordinates
* Roll, Pitch, Yaw 

Note : 
1. Code has beed tested for only GPU environments
2. Takes at least 16GB of GPU memory
