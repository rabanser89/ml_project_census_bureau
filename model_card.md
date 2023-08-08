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
Metrics in use are precision, recall, fbeta from the scikit-learn library. On the test data model performance is as follows:
- precision = 0.744
- recall = 0.631
- fbeta = 0.683

An example of model performence on data slices can be found in starter/slice_output.txt. In this case the education feature is used

## Ethical Considerations

There is no ethical concern to use this code.

## Caveats and Recommendations

Model inference can be obtained by a POST request to the endpoint https://census-bureau.onrender.com/inference. 
At the time of using the endpoint, it might have expired, but you can run the app locally by running
```
uvicorn main: app --reload
```
in the root domain. The documentation can then be found at http://127.0.0.1:8000/docs.


