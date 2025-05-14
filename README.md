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
pip install -r requirements.txt

For downloading the data
python prep_data.py --gf {path to dataset.zip}
python main.py --bs 32 --epochs 3 --no-log
