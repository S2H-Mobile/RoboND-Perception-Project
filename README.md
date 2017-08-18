# RoboND-Perception-Project
This is my solution of the [Perception Project](https://github.com/udacity/RoboND-Perception-Project) for the Udacity Robotics Nanodegree.

The challenge is a catkin ROS simulation environment where a PR2 robot must use its RGBD camera to perceive objects and place them in the appropriate dropbox.

- The file [writeup.md](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/writeup.md) documents the workflow and solution strategy.
- The ROS node [object_recognition.py](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/object_recognition.py) performs the object recognition task and creates ROS messages for the ``pick_place_routine`` service.

## Setup and Usage 
1. Follow the setup instructions of the [Udacity Perception Exercises](https://github.com/udacity/RoboND-Perception-Exercises) and the [Udacity Perception Project](https://github.com/udacity/RoboND-Perception-Project). After completing these steps you have a catkin workspace containing the packages ``sensor_stick`` and ``pr2_robot``.
2. Copy the script [object_recognition.py](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/object_recognition.py) to the directory ``/pr2_robot/scripts/`` in your catkin workspace.
3. Copy the script [features.py](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/object_recognition.py) to the directory ``/sensor_stick/src/sensor_stick/`` in your catkin workspace.
3. Follow the instructions in the workflow section to capture features, train a model and perform object recognition.

## Workflow
1. Train a classification model using features like color histograms and surface normals.
2. Run the object recognition pipeline and generate YAML files containing ROS messages.

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

![](world_2_object_recognition.png)