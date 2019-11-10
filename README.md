#### kaggle-pku
Team repo. for PKU Autnomous Driving Kaggle Competition.

To run the code -
1. If you are training locally ```Do pyhton train_model.py ```
2. If you are using the SSC cluster do - 
    * First navigate to the appropriate directory using ```cd /projectnb/cs542```
    * Then do ```cd {username}/kaggle-pku```
    * Then do `module load python3`
    * Then submit the job ```qsub scripts/qsub_job.template.sh```
    * To check the status on SCC server do ```qstat -u {username}```

Code explained -
1. This code uses [CenterNet](https://arxiv.org/pdf/1904.07850.pdf) to draw 3D bounding boxes for objects -

![task](images/task.JPG)

The quantities to be estimated are the -

* X, Y, Z co-ordinates
* Roll, Pitch, Yaw 

Note : 
1. Code has beed tested for only GPU environments
2. Takes at least 16GB of GPU memory
