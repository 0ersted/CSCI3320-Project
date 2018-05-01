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
Q: First, SVR accepts different kernel functions. They could
be one of linear, poly, rbf, sigmoid, precomputed, select one of them and state your reason in
prjreport.pdf. Second, epsilon and C are two critical parameters.Please state what role do they
play in the model, what value do you assign and why do you select these values
A: We selected rbf kernel since linear and poly could underfit, and when other parameters keep unchanged, the rbf kernel performs the best. 
   Epsilon is the margin of tolerance, the data within the region of tolerance would be neglected, so when epsilon is larger, the tolerance region would be larger, more data would be neglected, the accuracy of model tend to be lower, number of support vectors would be lower; When epsilon is lower, more of the data errors would be considered, but overfitting is more likely to happen.
   C is the cost, it represents the tolerance of error, when C is larger, the tolerance of error would be smaller, overfitting is more likely to happen; when C is smaller, tolerance of error would be larger, larger margin could be obtained, underfitting is more likely to happen; generally when number of noise points is large, C need to be smaller.
  I assigned 0.2 to epsilon and 27 to C since by cross-validation, I found that the model performs the best when epsilon=0.2 and C=27

#### 4.1.2 Gradient Boosting Regression Tree Model(GBRT)
Q: First, GradientBoostingRegressor accepts different loss
functions. They could be one of ls, lad, huber, quantile, select one of them and state your reason
in prjreport.pdf. Second, learning_rate, n_estimators and max_depth are three critical parameters.
Please state what role do they play in the model, what value do you assign and why do you select
these values
A: We selected huber as the loss function since when other parameters keep unchanged, the model would perform the best when "huber" is chosen to be the loss function. 
   Learning_rate is the learning rate of the procedure, a small learning rate generally lead to a better generalization error. 
   n_estimators is the number of boosting stages, usually when n_estimators is larger, the model performs better, and it's quite robust to over-fitting. 
   When learning_rate is small, a larger n_estimators is needed to ensure that the model is well-trained, but when n_estimator is larger, the time cost will be larger, so there is a trade-off. 
   Our strategy is to first set a large value for n_estimator, then tune the learning rate to achieve the best results. We got learning_rate=0.05, n_estimators=300

### 4.2 Predicting on Test Data
Q: Record your best result in the form (model_name, RMSE, Top_1, Top_3, Average_Rank) for both
SVR and GBRT model. Here, you are required to save your best result together with chosen parameters

A: SVR Model before normalization : RMSE =  19.104978238752828 ; Top_1 =  0.06860706860706861 ; Top_3 =  0.23492723492723494 ; Average_Rank =  6.704781704781705
   Gradient Boosting Regression Tree Model before normalization: RMSE =  32.699628017762414 ; Top_1 =  0.2494802494802495 ; Top_3 =  0.5550935550935551 ; Average_Rank =  3.9875259875259874
Q: Please try to normalize them and retrain your model to show whether normalizaiton improves the
result.
A:  After normalized and adjusting the parameters (for SVR: C=0.2, epsilon=0.1; for GBRT: loss='ls',learning_rate=0.012,n_estimators=1000,max_depth=1; we got the parameters by corss-validation), the RMSE, Top_1,Top3,Average_Rank become:
   SVR Model after normalization: RMSE =  1.7290500693496353 ; Top_1 =  0.1600831600831601 ; Top_3 =  0.38461538461538464 ; Average_Rank =  5.615384615384615
   Gradient Boosting Regression Tree Model after normalization: RMSE =  1.7519266975943029 ; Top_1 =  0.37422037422037424 ; Top_3 =  0.6361746361746362 ; Average_Rank =  3.501039501039501
   The Result improved after normalization

##  5 Betting Strategy


## 6 Visualization
### 6.1 Line Chart of Recent Racing Result

### 6.2 Scatter Plot of Win Rate and Number of Wins

### 6.3 Pie Chart of the Draw Bias Effect

### 6.4 Bar Chart of the Feature Importances

### 6.5 Visualize SVM