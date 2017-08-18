# Project: Perception Pick & Place

[//]: # (Image References)

[screenshot_world_1]: ./world_1_object_recognition.PNG
[screenshot_world_2]: ./world_2_object_recognition.PNG
[screenshot_world_3]: ./world_3_object_recognition.PNG

## Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  See the example `output.yaml` for details on what the output should look like.  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

## Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

# [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points

### Exercise 1, 2 and 3 Pipeline
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

#### 3. Complete Exercise 3 Steps. Features extracted and SVM trained. Object recognition implemented.
To generate the training sets, I modified the objects list in the ``capture_features.py`` script with the items from ``pick_list_3.yaml``, which contains all object models:
```
    models = [\
       'biscuits',
       'soap',
       'soap2',
       'book',
       'glue',
       'sticky_notes',
       'snacks',
       'eraser']
```

For engineering the features I assumed (following the lectures) that an object is characterized by the following properties:
- the distribution of color channels,
- the distribution of surface normals.
To extract characteristic features from these data distributions I organized them in histograms and tuned the following parameters
- number of poses per object,
- usage of either RGB or HSV color space,
- number of bins in a histogram,
in order to optimize the performance of the SVM classifier.

For example, the table below shows data for several capture cycles where 8 poses where taken with varying bin sizes and color spaces for the color histograms.
 
color space / bin size | 32    | 64   | 128
---                    | ---   | ---  | ---
RGB                    | 0.625 | -    | -
HSV                    | 0.906 | -    | -

The accuracy scores for RGB color space were higher than for HSV color space for 32 and 64 bins. The accuracy decreased with increasing bin size.

For the number of poses per object, I tried the values 8, 10, 40, 100. For 10 and 40 the models seemed to overfit since the cross-validation score improved but the real world accuracy decreased.

Finally I trained a model with 100 poses per object. The histograms have 32 bins per channel. The color histograms are in  HSV space with range (0, 256) and the surface normal histograms have a range of (-1, 1). This model reached an accuracy score of 0.99 in cross-validation and successfully detected all objects in the test scenes 1, 2 and 3.

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

### Results
In the final configuration, the [perception pipeline](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/object_recognition.py) recognized all objects in all the test scenes. The screenshots below show clippings of the RViz window subscribed to the ``/pcl_objects`` publisher. The objects in the scene are labeled with the predicted label.

#### Scene 1
See the file output_1.yaml and the screenshot below.
![Recognized objects for scene 1.][screenshot_world_1]

#### Scene 2
See the file output_2.yaml and the screenshot below.
![Recognized objects for scene 2.][screenshot_world_2]

#### Scene 3
See the file output_3.yaml and the screenshot below.
![Recognized objects for scene 3.][screenshot_world_3]

#### Improvements
Since the performance of the perception depends on a number of parameters, the main improvements can be made by tuning those parameters.

- The parameters of the image pipeline need to be adjusted to the hardware in use and the application (for example adapt to changing region of interest and object distribution in the scene).
- The selection and quality of the features used to train the SVM.
- The machine learning model. Which kernel type fits best to the features, linear or RBF. 


