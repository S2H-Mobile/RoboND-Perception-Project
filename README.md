# RoboND-Perception-Project

[//]: # (Image References)

[screenshot_world_3]: ./world_3_object_recognition.PNG

This is my solution of the [Perception Project](https://github.com/udacity/RoboND-Perception-Project) for the Udacity Robotics Nanodegree.

The challenge is a catkin ROS simulation environment where a PR2 robot must use its RGBD camera to perceive objects and place them in the appropriate dropbox.

## Contents
- The file [writeup.md](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/writeup.md) documents the solution strategy.
- The ROS node [object_recognition.py](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/object_recognition.py) performs the object recognition task and creates ROS messages for the ``pick_place_routine`` service.
- The script [capture_features.py](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/capture_features.py) is used to generate feature data used in [train_svm.py](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/train_svm.py) for training a SVM model.
- The script [features.py](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/features.py) defines the features used to characterize the objects in the scene.

## Setup and Usage 
1. Follow the setup instructions of the [Udacity Perception Exercises](https://github.com/udacity/RoboND-Perception-Exercises) and the [Udacity Perception Project](https://github.com/udacity/RoboND-Perception-Project). After completing these steps you have a catkin workspace containing the packages ``sensor_stick`` and ``pr2_robot``.
2. Copy the script [object_recognition.py](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/object_recognition.py) to the directory ``/pr2_robot/scripts/`` in your catkin workspace.
3. Copy the script [features.py](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/object_recognition.py) to the directory ``/sensor_stick/src/sensor_stick/`` in your catkin workspace.
3. Follow the instructions in the workflow section to capture features, train a model and perform object recognition.

## Workflow
1. Train a classification model using features like color histograms and surface normals.
2. Adjust the parameters of the object recognition pipeline.
3. Run the pipeline for the test scenes and generate YAML files containing ROS message ROS messages for the ``pick_place_routine`` service.
4. Optionally, perform the pick and place operation with the PR2 robot arm.

### Capture Features and Train Model
- Set up the training environment with ``roslaunch sensor_stick training.launch``.
- The features are defined by the functions ``compute_color_histograms()`` and ``compute_normal_histograms()`` in ``/sensor_stick/src/sensor_stick/features.py``.
- Run the feature capturing script with ``rosrun sensor_stick capture_features.py``. When it finishes running you have a ``training_set_<num_poses>.sav`` file.
- Run the training script with ``rosrun sensor_stick train_svm.py``. When it finishes running you have a ``model_<num_poses>.sav`` file.

### Run Object Recognition
- Add your object recognition code to your perception pipeline.
Test with the actual project scene to see if your recognition code is successful.
- Modify the file ``/pr2_robot/launch/pick_place_project.launch`` to select the test scene.
- Launch the RViz environment with ``roslaunch pr2_robot pick_place_project.launch``.
- Run the object recognition script ``rosrun pr2_robot object_recognition.py``. Markers will show up in RViz like so:

![Recognized objects for scene 3.][screenshot_world_3]