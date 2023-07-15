# Model Card

## Model Details

The model used in this project is a Random Forest Classifier of scikit-learn whit default hyperparameter. 

## Intended Use

The model predicts whether people have low salary(<=50k) or high salary (>50k).

## Training Data and Evaluation Data
The census data are obtained from here: https://archive.ics.uci.edu/dataset/20/census+income.

The dataset has a length of 32561 and is splitted into training data and test data with ratio 0.2. Ther are 14 features in the dataset 

To use the dataset for model training the data is proccessed by using one hot encoding for the features and a label binarizer for the labels


## Metrics
_Please include the metrics used and your model's performance on those metrics._
Metrics in use are precision, recall, fbeta. On the test data model performance is as follows:
- precision = 0.744
- recall = 0.631
- fbeta = 0.683

An example of model performence on data slices can be found in slice_output.txt. In this case the education feature is used

