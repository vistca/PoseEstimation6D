# PoseEstimation6D

Start with next NN and look into how to crop images (selecting bbox) to insert into the pose predictor. How to process images from BBox to 6D Pose. Phase 3
- Erik
- Carl

Evaluation part. Implement mAP metric in phase 2
- Vilho

Implement YOLO. Start training in the phase 2
- Felix

# How to run
After cloaning the project run the following commands in the terminal:
```
pip install -r requirements.txt
```

For downloading the data and placing it in the correct path run
```
python prep_data.py --gf {path to dataset.zip}
```

Finally, start training and testhing through:
```
# Without logging 
python main.py --bs 32 --epochs 3 --no-log

# With logging 
python main.py --bs 32 --epochs 3 --wb {your key}
```

# Running pose/main,py
python -m pose.main --bs 32 --epochs 3 --no-log --mod "res" --test True
```

Current state of the project:

 - We have the Faster R-CNN but it has not trained it
 

Wish list:

 - Being able to see the bounding boxes

 - Normalizeing the data, maybe?

 - Step 3 can be worked on indepentedly of step 2

 - Add validation set and retrieve validation accuracy


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


TODO:
 - Add object id as another input to the classifier of the pose estimator

 - Normalize the images with std and mean???

 - Add that we separate the checkpoints for Faster R-CNN and EfficientNet
