# PoseEstimation6D

Start with next NN and look into how to crop images (selecting bbox) to insert into the pose predictor. How to process images from BBox to 6D Pose. Phase 3
- Erik
- Carl

Evaluation part. Implement mAP metric in phase 2
- Vilho

Implement YOLO. Start training in the phase 2
- Felix

# How to run

### Prep project
After cloaning the project run the following commands in the terminal:
```
pip install -r requirements.txt
```

For downloading the data and placing it in the correct path run
```
python prep_data.py --gf {path to dataset.zip}
```

## Running entire project
WIP. TO be added

## Running individual parts

### Run phase 2 separately
To train and run phase 2, i.e just retrieving the the 2d bounding boxes run the following

```
python -m 2dBox.main --{args}

The arguments include:
- lr:  learning rate
- bs:  batch size
- epochs:  the number of epochs
- optimizer:  the name of the optimizer
- scheduler:  the name of the scheduler
- wb:   if wandb should be used, else specify --no-log
- lm:   name for loading model 
- sm:   name for saving model
- tr:   how much of the model that is trainable
- fm:   fasterRcnn model name
- test:     if testing should be carried out
```

### Run phase 3 separately
To train and run phase 3, i.e estimating the 6d pose from rgb images run the following

```
python -m pose.main --{args}

The arguments include:
- lr:  learning rate
- bs:  batch size
- epochs:  the number of epochs
- optimizer:  the name of the optimizer
- scheduler:  the name of the scheduler
- wb:   if wandb should be used, else specify --no-log
- lm:   name for loading model 
- sm:   name for saving model
- mod:   the name of the model that should be used  
- test:   if testing should be carried out

```

### Run phase 4 separately
To train and run phase 4, i.e estimating the 6d pose from rgb and depth images run the following

```
python -m pose.main --{args}

The arguments include:
- lr:  learning rate
- bs:  batch size
- epochs:  the number of epochs
- optimizer:  the name of the optimizer
- scheduler:  the name of the scheduler
- wb:   if wandb should be used, else specify --no-log
- lm:   name for loading model 
- sm:   name for saving model
- mod:   the name of the model that should be used  
- test:   if testing should be carried out
```



Questions for the supervisor:

 - What kind of model should be used for the first pose estimation step

 - What does the rotation matrix and position represent, are they some relative orientation?

 - Variable size of inputs to the pose estimator, pixel-wise embeddings

 - What should be the acctual loss function, intersection over ... positional loss...

 
Answers from the supervisor session:

 - CNN, we can choose the model.

 - It is the object's rotation and orientation relative to the camera

 - Stretch and squeeze it to a fixed size (paddings could be used)

 - We could just use the MSE but if we want to we could define a custom loss function

 - Input the rations of the image size transfomration to the fully connected layer of the model.

