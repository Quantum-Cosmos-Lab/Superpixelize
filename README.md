# Superpixelize
The tools to reduce the image binary classification training data for SVM use. It is based on superpixel routine and developed for use in [38-Clouds dataset](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images).

## Structure

The main file is [reduce.ipynb](https://github.com/Quantum-Cosmos-Lab/Superpixelize/blob/main/reduce.ipynb) in which we show a basic usage of the script. It assumes that the Superpixelize directory is inside the main 38-Clouds directory.

The project utilizes the slightly modified 38-Clouds dataset class, originaly created by Mauricio Cordeiro [38-Cloud-Data preparation](https://www.kaggle.com/code/cordmaur/38-cloud-data-preparation). The tools to handle dataset scenes are present in [cloud_classification_data.py file](https://github.com/Quantum-Cosmos-Lab/Superpixelize/blob/main/cloud_classification_data.py).

The tools for superpixel data reduction are present in [superpixelize_tools.py](https://github.com/Quantum-Cosmos-Lab/Superpixelize/blob/main/superpixelize_tools.py)

## How is data processed?

For each scene we reduce each patch. 
The reduction consists of making a data point prototype from the statistical measures of each segment of divided patch.
The segments are obtained by dividing each RBG patch into N superpixels with [SLIC](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#slic) algorithm.

The 6 statistical measures we derive in the current version of the tool are:
- Median
- Mean
- IQR
- Max
- Min
- StdDev

Each of the above measures is applied to each of image bands.
On default in 38-Clouds dataset we have 4 bands (R, G, B, Nir), which results in 4 spectral bands * 6 statistical measures = 24 features.
Additionally we add labels to each superpixel by taking an average of points label in the corresponding segment. 
Therefore the resulting dataset consists of 25 columns.
