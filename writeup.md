# Project: Perception Pick & Place

[//]: # (Image References)

[screenshot_world_1]: ./world_1_object_recognition.PNG
[screenshot_world_2]: ./world_2_object_recognition.PNG
[screenshot_world_3]: ./world_3_object_recognition.PNG

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points

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
 
color space / bin size | 32    | 64    | 128
---                    | ---   | ---   | ---
RGB                    | 0.625 | 0.797 | 0.641
HSV                    | 0.906 | 0.984 | 0.922

The accuracy scores for HSV color space are consistently higher than for HSV color space across all bin sizes. A bin size of 64 gives the highest accuracy score, followed by bin sizes 128 and 32. Later on it turned out that the derived model for bin size 32 and HSV color space performed best on the test scenarios, so I used these values.

For the number of poses per object, I tried the values 8, 10, 40, 100. For 10 and 40 the models seemed to overfit since the cross-validation score improved but the real world accuracy decreased.

Finally I trained a model with 100 poses per object. The histograms have 32 bins per channel. The color histograms are in  HSV space with range (0, 256) and the surface normal histograms have a range of (-1, 1). This model reached an accuracy score of 0.99 in cross-validation and successfully detected all objects in the test scenes 1, 2 and 3.

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

In the final configuration, the [perception pipeline](https://github.com/S2H-Mobile/RoboND-Perception-Project/blob/master/scripts/object_recognition.py) recognized all objects in all the test scenes. The screenshots below show clippings of the RViz window subscribed to the ``/pcl_objects`` publisher. The objects in the scene are labeled with the predicted label.

##### Results 
###### Scene 1
See the file output_1.yaml and the screenshot below.

![Recognized objects for scene 1.][screenshot_world_1]

###### Scene 2
See the file output_2.yaml and the screenshot below.

![Recognized objects for scene 2.][screenshot_world_2]

###### Scene 3
See the file output_3.yaml and the screenshot below.

![Recognized objects for scene 3.][screenshot_world_3]

##### Improvements
Since the performance of the perception depends on a number of parameters, the main improvements can be made by tuning those parameters.

- The parameters of the image pipeline need to be adjusted to the hardware in use and the application (for example adapt to changing region of interest and object distribution in the scene).
- The selection and quality of the features used to train the SVM.
- The machine learning model. Which kernel type fits best to the features, linear or RBF. 

## Extra Challenges: Complete the Pick & Place

Coming soon!


