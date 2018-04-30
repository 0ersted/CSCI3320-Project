# CSCI3320 PROJECT REPORT

## 2 The dataset and pre-processing
### 2.2 Data preprocessing
#### 2.2.3 Indices and features for horses, jockeys and trainers

Numer of horses:  2155
Numer of jockeys:  105
Numer of trainers:  93

## 3 Classification
### 3.1 Training Classifiers in Scikit-Learn
#### 3.1.1 Logistic Regression

#### 3.1.2 Naïve Bayes
Here we choose **GaussionNB** classifier using Gaussian distribution to estimate the likelihood. The reason we choose it is that first we assume each feature holds Gaussian distribution. Also, we have some features in real value(continuous case).


#### 3.1.3 SVM

#### 3.1.4 Random Forest

### 3.3 Evaluation of Predictions

### 3.4 Writing A Report
* **Q**: What are the characteristics of each of the four classifiers?
  **A**: We analize each chasifiers one by one.
    * **Logistic Regression**: It is easy and fast. And it works reletively good when we have a large amount of data. It don't need to consider any assumption among samples.
    
    * **Naïve Bayes**: Naïve Bayes is the fastest classifier in these four. It is really simple to implement. However, the probability assumptions that the features are independent may not hold in reality. If the somes features are dependent, the classifier may not work so well.

    * **SVM**: The classifier really depends on a good kernel function. It is tedious and costy to choose a good one but is expected to work well if a good kernel is found. And the training process is very long because we need a huge amount of computation.
    
    * **Random Forest**: It is a kind of ensemble learning taking the advantages of decision tree. It is easy to explain and has a good performance as well as low risk of overfitting as it generates each decision tree on random.

* **Q**: Different classification models can be used in different scenarios. How do you choose classification models for different classification problems? Please provide some examples.
  **A**:
  * **Logistic Regression**: It is suitable in most scenarios as a base line to evaluate other models.
    
  * **Naïve Bayes**: It is considered when features seem like independent. Online learning is also a good reason so the model can be improved as the number of samples increaces. So it works well in jobs like email spam. 

  * **SVM**: It works well when the sample set and feature set  are not too large so the training time wouldn't be too long. If we have a good kernel, SVM usually performs well.
    
  * **Random Forest**: We can consider random forest when number of features are not too huge and we want reasonable good results. It can be tried first without fitting parameters.
  
* **Q**: How do the cross validation techniques help in avoiding overfitting?
  **A**: By doing cross validation, we can get a sense on the prediction score and results without touching any testing data. It will help us to find suitable hyper parameters and compare each model.

* **Q**:How do you choose evaluation metrics for imbalanced datasets according to the class distribution? Please give your understanding and provide some examples.
  **A**:  

## 4 Regression
### 4.1 Training Regression Model in Scikit-Learn
#### 4.1.1 Support Vector Regression Model(SVR)

#### 4.1.2 Gradient Boosting Regression Tree Model(GBRT)

### 4.2 Predicting on Test Data

##  5 Betting Strategy


## 6 Visualization
### 6.1 Line Chart of Recent Racing Result

### 6.2 Scatter Plot of Win Rate and Number of Wins

### 6.3 Pie Chart of the Draw Bias Effect

### 6.4 Bar Chart of the Feature Importances

### 6.5 Visualize SVM