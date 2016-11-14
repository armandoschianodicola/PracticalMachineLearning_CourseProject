# Practical Machine Learning Course Project
Armando Schiano di Cola  
13 Novembre 2016  

## Introduction
***

Nowadays, several wearable devices like Jawbone Up, Nike FuelBand, and Fitbit provide the possibility to obtain data about personal activity in a relatively inexpensive way. This data could then be used to make inference about the way the physical activities were performed.
Given these premises, the goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

More information about the data is available from the website here: http://groupware.les.inf.puc-rio.br/har 


## Environment Set-Up 
***

As a first step, we will set up our environment and load the libraries


```r
# clean up the environment
rm(list=ls())

# load libraries
library(ggplot2)
library(lattice)
library(foreach)
library(iterators)
library(caret)
library(RANN)
library(parallel)
library(doParallel)
library(plyr)
library(survival)
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```r
library(gbm)
```

```
## Loading required package: splines
```

```
## Loaded gbm 2.1.1
```


```r
set.seed(1987)
```

## Getting data
***

Import the data


```r
# get data
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

trainImport <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testImport <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
```

## Preprocessing
***

Now, let's do some preprocessing of the data. In this phase we will partition the entire dataset into a training and a test set, with a proportion of 80/20


```r
# partition the data
inTrain <- createDataPartition(trainImport$classe, 
                               p=0.8, 
                               list=FALSE
                               )
training <- trainImport[inTrain, ]
testing <- trainImport[-inTrain, ]

# remove useless variables (e.g. "user name")
training <- training[-c(1:2)]
```

and then let's remove the variable we are trying to predict, namely the "classe" variable, from the training set


```r
trainImport <- subset(trainImport, select = -c(classe))
```

To improve the process of model fitting, we need to remove some variables that have near zero-variance or that have a lot of NAs. For the latter, we use a threshold of 80% of NAs for the predictors to be excluded.  


```r
# remove near zero variance variables
nzv_var <- nearZeroVar(training, saveMetrics=TRUE)

training <- training[, nzv_var$nzv == FALSE]

na_sums <- apply(training, 2, FUN = function(x) sum(is.na(x)))
na_perc <- apply(training, 2, FUN = function(x) (sum(is.na(x)) / length(x)))

# remove variables with more than 80% NAs 
training <- training[ , !(names(training) %in% names(which(na_perc > 0.80)))]
testing <- testing[ , !(names(testing) %in% names(which(na_perc > 0.80)))]
```


```r
# clean memory to free up space
gc(verbose = FALSE)
```

```
##           used (Mb) gc trigger  (Mb) max used  (Mb)
## Ncells 1600207 85.5    2637877 140.9  1829905  97.8
## Vcells 5791493 44.2   17256449 131.7 21569283 164.6
```

```r
rm(testImport, trainImport)
```



## Model Fitting
***

By now we should have our dataset ready to put it into an algorithm of statistical learning. 

Before that, we first use the function `trainControl` to compute parameters of resampling. In our case, we decide to do a 10-fold cross validation. This parameter will be inserted in our model


```r
fitControl <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE)
```

Then we finally choose the algorithm of statistical learning for our dataset. Since the predicted outcome is a factor variable, we can use a classification technique known as `Boosted Tree` model. The Boosted Tree model is a machine learning technique which creates a prediction model in the form of an "ensemble" of weak prediction models, typically decision trees. The Boosted tree model uses the same approach as a single tree, but sums up many weighted tree models over each boosting iteration.


```r
modelFitGbm <- train(classe ~ .,
                     data = training,
                     method = "gbm",
                     trControl = fitControl,
                     verbose = FALSE
                     )
```



We can then predict the outcome on the testing set using our model and analyze the results


```r
pred_gbm <- predict(modelFitGbm,
                    testing
                    )

confusionMatrix(pred_gbm,
                testing$classe)
```

```
## $positive
## NULL
## 
## $table
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    0    0    0
##          B    0  756    1    0    0
##          C    0    2  677    2    0
##          D    0    0    6  640    1
##          E    0    0    0    1  720
## 
## $overall
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9964313      0.9954860      0.9940196      0.9980476      0.2844762 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN 
## 
## $byClass
##          Sensitivity Specificity Pos Pred Value Neg Pred Value Precision
## Class: A   1.0000000   0.9996437      0.9991047      1.0000000 0.9991047
## Class: B   0.9960474   0.9996839      0.9986790      0.9990524 0.9986790
## Class: C   0.9897661   0.9987651      0.9941263      0.9978408 0.9941263
## Class: D   0.9953344   0.9978659      0.9891808      0.9990842 0.9891808
## Class: E   0.9986130   0.9996877      0.9986130      0.9996877 0.9986130
##             Recall        F1 Prevalence Detection Rate
## Class: A 1.0000000 0.9995522  0.2844762      0.2844762
## Class: B 0.9960474 0.9973615  0.1934744      0.1927097
## Class: C 0.9897661 0.9919414  0.1743564      0.1725720
## Class: D 0.9953344 0.9922481  0.1639052      0.1631405
## Class: E 0.9986130 0.9986130  0.1837879      0.1835330
##          Detection Prevalence Balanced Accuracy
## Class: A            0.2847311         0.9998219
## Class: B            0.1929646         0.9978657
## Class: C            0.1735916         0.9942656
## Class: D            0.1649248         0.9966001
## Class: E            0.1837879         0.9991504
## 
## $mode
## [1] "sens_spec"
## 
## $dots
## list()
## 
## attr(,"class")
## [1] "confusionMatrix"
```

Finally, we have to estimate the "out of sample error" of our model. The out of sample error is the error that we get on a data set different from our training set. In our case it is the "testing" set.
Out of sample errors could be estimated through a wide range of values, some of which are indicated in the "overall" section of the confusion matrix. For our purpose, we will use the `Accuracy` value, which is very high for our model (0.9964313). If we look further at the table we will see clearly that the prediction and the reference class almost overlap, as shown below:


```r
confusionMatrix(pred_gbm, testing$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    0    0    0
##          B    0  756    1    0    0
##          C    0    2  677    2    0
##          D    0    0    6  640    1
##          E    0    0    0    1  720
```

we can thus conclude that our model fits well on the dataset. 
