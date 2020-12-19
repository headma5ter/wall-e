# TRI-AD
Contains a k-means algorithm that can be run on any N-dimensional dataset 
in order to classify data into k clusters. Data included is the classic Iris 
species determination dataset.


### How to use
If the user wants the log written to file, it must be included as a environment 
variable (`LOG_PATH`); otherwise it is written to console.

```commandline
cd path\to\module\dialup\src
python -m iris.main
```


## Configuration file
All user-input variables are held within the config file (`src\triad\config.ini`). 
The following options can be adjusted in order to configure either the algorithm 
(located under `[settings]`) or various file paths (located under `[paths]`).

### settings
  - `stage`: either `training` or `testing` 
  - `clusters`: k clusters to use in algorithm
  - `norm`: how to calculate distance (`L2` or `L1`)
  - `visualize`: view plots (boolean)
  - `serialize`: save data to pickle/json (boolean)
  - `save`: save results and plot (boolean) 
  
### paths
  - `training`: training data (REQUIRED)
  - `testing`: testing data (REQUIRED)
  - `centroids`: pickle file for centroids data
  - `mapping`: json file for mapping cluster to species
  - `results`: results file after running over training data
  - `plot`: path for plot (prefixed with appropriate `stage`)


## Pipeline
The system runs in two different ways, depending on which `stage` is being run:

### Training
The training data is read in, run through the algorithm, then saved to file. The data 
saved to file is the centroid data (the k-by-N matrix indicating the centers of each 
cluster) and mapping data (a dictionary which maps each k cluster ID to Iris species). 
A plot is also created, showing the classification of each data point.

### Testing
The testing data is read in and run through the algorithm using the serialized 
centroids from the training data as starting points. The results file and plot are then 
written to file. Statistics are contained within the log.

NOTE: if no centroid data is 
found/provided, it will run the algorithm from scratch, which could lead to poor 
performance.


## Contact
For questions, please reach out to Kevin Ludwig (ludwigkg@gmail.com).
