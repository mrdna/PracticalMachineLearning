Practical Machine Learning Pediction Assignment
========================================================

This assignment is to use data collected from activity sensors to predict if a dumbell lift was performed correctly or in one of 4 error classes. The data are supplied this project come from this source: http://groupware.les.inf.puc-rio.br/har. The class of exercise is listed in the "classe" variable as A, B, C, D or E.

To summarize my approach, I first explored the data to reduce the number of prediction variables, removed highly correlated variables and built a model using the random forest method. I calculated the out of sample error rate using a testing set of data that was not used to build the model.

## Data processing:
The first step I took was to read the data set from my working directory and separate the data 60/40 into a training and test set.


```r
pmlData <- read.csv("pmlTraining.csv")
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.1
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.1
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
set.seed(1357)
inTrain <- createDataPartition(y=pmlData$classe, p=0.6, list=FALSE)
training <- pmlData[inTrain,]
testing <- pmlData[-inTrain,]
```

Calling summary on the data showed that many columns have 19216 out of 19622 entries that are NA. I determined that these would not be useful as predictors so removed them from the data set. 

According to the publication (http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201), which was linked in the reference data set, 8 features are calculated and not raw measured data: mean, variance, standard deviation, minimum, maximum, amplitude, kurtosis and skewness. Columns related to these calculated features as well as relating to user name and time variables were also removed (if not already removed during the NA removal step). 

The resulting data set contains 52 predictor columns. The classe identifier column is column 53. All columns other than the classe column were converted to numeric values.

```r
noNA <- training[,colMeans(is.na(training)) == 0] 
noNA3 <- noNA[,-grep('time|X|user|window|max|min|amplitude|kurtosis|skewness',names(noNA))]
cols.num <- c(1:52)
noNA3[cols.num] <- sapply(noNA3[cols.num],as.numeric)
```
I then checked for variables with high correlation.


```r
corTrain <- cor(noNA3[,-53])
library(corrplot)
corrplot(corTrain, method = "circle")
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3.png) 
Predictors that have a high correlation are indicated by the dark coloring in the plot. This indicates preprocessing would be useful to remove redundant predictors. Preprocess using PCA.


```r
preRF <- preProcess(noNA3[,-53], method="pca")
trainRF <- predict(preRF, noNA3[,-53])
```
## Train prediction model
A random forest method was used for model building.

```r
fitRF <- train(noNA3$classe ~ ., method="rf", data=trainRF)
fitRF
```

```
## Random Forest 
## 
## 11776 samples
##    24 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    1.0       0.9    0.004        0.005   
##   13    0.9       0.9    0.004        0.005   
##   25    0.9       0.9    0.005        0.006   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```
## Apply model to the test set data
Before running the test set data through the model the same filtering and preprocessing that was done to the training data must be aplied to the test set.

```r
noNAtest <- testing[,colMeans(is.na(training)) == 0] 
noNAtest3 <- noNAtest[,-grep('time|X|user|window|max|min|amplitude|kurtosis|skewness',names(noNAtest))]
noNAtest3[cols.num] <- sapply(noNAtest3[cols.num],as.numeric)
preTestRF <- predict(preRF, noNAtest3[,-53])
```
A confusion matrix can be used to see the model accuracy for the training set data.

```r
confusionMatrix(testing$classe, predict(fitRF, preTestRF))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2209    9    6    7    1
##          B   32 1460   25    0    1
##          C    2   17 1338    6    5
##          D    6    3   78 1196    3
##          E    0   18    8   12 1404
## 
## Overall Statistics
##                                         
##                Accuracy : 0.97          
##                  95% CI : (0.965, 0.973)
##     No Information Rate : 0.287         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.961         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.982    0.969    0.920    0.980    0.993
## Specificity             0.996    0.991    0.995    0.986    0.994
## Pos Pred Value          0.990    0.962    0.978    0.930    0.974
## Neg Pred Value          0.993    0.993    0.982    0.996    0.998
## Prevalence              0.287    0.192    0.185    0.156    0.180
## Detection Rate          0.282    0.186    0.171    0.152    0.179
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.989    0.980    0.957    0.983    0.994
```
The accuracy of this model is 0.9695. Error can be calculated as 1-Accuracy, so the out of sample error estimate using the test set data is 0.0305.  Appling the model to the 20 test samples provided in the assignment resulted in 19 or 20 correctly prediced exercise classes. This corresponds to an out of sample error rate of 0.05.
