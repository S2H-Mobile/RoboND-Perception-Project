# Project: Perception Pick & Place

[//]: # (Image References)

[screenshot_world_1]: ./world_1_object_recognition.PNG
[screenshot_world_2]: ./world_2_object_recognition.PNG
[screenshot_world_3]: ./world_3_object_recognition.PNG
[normalized_confusion_matrix]: ./normalized_confusion_matrix.PNG

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points

### Exercise 1, 2 and 3 Pipeline
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
My image recognition pipeline:

1. Convert the point cloud which is passed in as a ROS message to PCL format.
2. Filter out the camera noise with the PCL statistical outlier filter. The adjustable parameters are the number ``k`` of neighbouring pixels to average over and the outlier threshold ``thr = mean_distance + x * std_dev``. I used the RViz output image to tune these parameters judging by the visual output. I found that the values ``k = 8`` and ``x = 0.3`` performed best at removing as much noise as possible without deleting content.
3. Downsample the image with a PCL voxel grid filter to reduce processing time and memory consumption. The adjustable parameter is the leaf size which is the side length of the voxel cube to average over. A leaf size of 0.005 seemed a good compromise between gain in computation speed and resolution loss.
4. A passthrough filter to define the region of interest. This filter clips the volume to the specified range. My region of interest is within the range ``0.6 < z < 1.1`` and ``0.34 < x < 1.0`` which removes the dropboxes.
5. RANSAC plane segmentation in order to separate the table from the objects on it. A maximum threshold of 0.01 worked well to fit the geometric plane model. The cloud is then segmented in two parts the table (inliers) and the objects of interest (outliers). The segments are published to ROS as ``/pcl_table`` and ``/pcl_objects``. From this point onwards the objects segment of the point cloud is used for further processing.

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
In order to detect individual objects, the point cloud needs to be clustered.

Following the lectures I applied Euclidean clustering. The parameters that worked best for me are a cluster tolerance of 0.02, a minimum cluster size of 40 and a maximum cluster size of 4000. The optimal values for Euclidean clustering depend on the leaf size defined above, since the voxel grid determines the point density in the image.

The search method is [k-d tree](http://pointclouds.org/documentation/tutorials/kdtree_search.php), which is appropriate here since the objects are well separated the x and y directions (e.g seen when rotating the RViz view parallel to z-axis).

The clusters are colored for visualization in RViz, the corresponding ROS subject is ``/pcl_cluster``.

The next part of the pipeline handles the actual object recognition using machine learning.

#### 3. Complete Exercise 3 Steps. Features extracted and SVM trained. Object recognition implemented.
To generate the training sets, I added the items from ``pick_list_3.yaml``, which contains all object models, to the [capture_features.py](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/capture_features.py) script:
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

To extract characteristic features from these data distributions I organized them in histograms. My implementation for computing the histograms follows the lectures and can be found in [features.py](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/features.py).

I tuned the following parameters in order to optimize the performance of the SVM classifier:
- number of poses per object,
- usage of either RGB or HSV color space,
- number of bins in a histogram.


For example, the table below shows data for several capture cycles where 8 poses where taken with varying bin sizes and color spaces for the color histograms.
 
color space / bin size | 32    | 64    | 128
---                    | ---   | ---   | ---
RGB                    | 0.625 | 0.797 | 0.641
HSV                    | 0.906 | 0.984 | 0.922

The accuracy scores for HSV color space are consistently higher than for RGB color space across all bin sizes. A bin size of 64 gives the highest accuracy score, followed by bin sizes 128 and 32. Later on it turned out that the derived model for bin size 32 and HSV color space performed best on the test scenarios, so I used these values.

For the number of poses per object, I tried the values 8, 10, 40, 100. For 10 and 40 the models seemed to overfit since the cross-validation score improved but the real world accuracy decreased.

Finally I trained a model with 100 poses per object. The histograms have 32 bins per channel. The color histograms are in  HSV space with range (0, 256) and the surface normal histograms have a range of (-1, 1). This model reached an accuracy score of 0.99 in cross-validation and later on successfully detected all objects in the test scenes 1, 2 and 3. The following figure shows the normalized confusion matrix produced by the ``train_svm.py`` script.

![Normalized confusion matrix for training the model.][normalized_confusion_matrix]

The trained model is then used in the ``pcl_callback`` routine to do the inception on the list of clusters from part 2. This means for each cluster in the list the features are calculated and then the classifier predicts the most probable object label. These labels are published to ``/object_markers`` and the detected objects of type ``DetectedObject()`` are published to ``/detected_objects``.

Then the ``pr2_mover`` routine is called in order to generate a ROS message for each detected object. Within ``pr2_mover``, I first check the accuracy of the prediction. I check if the number of detected objects matches the number of items in the pick list. Then I match the detected labels to the ground truth labels taken from the pick list for the test scene.

I found that often the objects were labeled correctly, but the sort order differs from the pick list. So I sort the detected objects list in the order of the pick list.

For the resulting ``sorted_objects`` list I create the ROS message data for the ``pick_place_routine`` service. First I compute the list of point cloud centroids by taking the mean in the x, y and z direction of the points. Then I generate the list of ``dropbox_groups`` to indicate the target dropbox, by matching each object with the pick list by label.

For creating the ROS meassages I convert the NumPy data types into native types, for example the centroids:

```scalar_centroid = [np.asscalar(element) for element in np_centroid]```

Then I use the predefined methods to save the resulting list of dictionaries to a YAML file and call the ``pick_place_routine`` service.

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

##### Results
In the final configuration, the [perception pipeline](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/object_recognition.py) recognized all objects in all the test scenes. The screenshots below show clippings of the RViz window subscribed to the ``/pcl_objects`` publisher. The objects in the scene are labeled with the predicted label.
 
###### Scene 1
See the file [output_1.yaml](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/output/output_1.yaml) and the screenshot below.

![Recognized objects for scene 1.][screenshot_world_1]

###### Scene 2
See the file [output_2.yaml](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/output/output_2.yaml) and the screenshot below.

![Recognized objects for scene 2.][screenshot_world_2]

###### Scene 3
See the file [output_3.yaml](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/output/output_3.yaml) and the screenshot below.

![Recognized objects for scene 3.][screenshot_world_3]

##### Improvements
The performance of the pipeline depends on the perception parameters and the machine learning model. So the main improvements can be made in the following three categories.

- Parameter tuning. Adjust perception parameters to the camera hardware in use and the application. For example, adapt to changing region of interest and object distribution in the scene.
- Feature engineering. Do a grid search to find the parameters that optimize the accuracy score,
- Selecting the machine learning model. For example, change the Kernel type of the SVM (in this project the linear kernel performs better than RBF). Apply more sophisticated deep learning techniques.

## Extra Challenges: Complete the Pick & Place

Coming soon!


