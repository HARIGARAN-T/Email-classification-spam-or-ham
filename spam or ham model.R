data <- read.csv("demo spam.csv")

# import library
library(caret)
library(C50)                      #...decision tree...#
library(caretEnsemble)           #...to improve performance of classifier...#
library(kernlab)
library(MLeval)

str(data)
dim(data)
data$class <- as.factor(data$class)

# splitting the data to train and test data set 
set.seed(500)
size <- floor(0.66*nrow(data))
set.seed(15)
trainsize <- sample(seq_len(nrow(data)),size = size)
traindata <- data[trainsize,]
testdata <- data[-trainsize,]


#  machine learning model


# control parameter

control<-trainControl(method = "cv", summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)

# logistic regression

modellog <- train(class~.,methods="glm",data = traindata,trControl = control)
pred_log <- predict(modellog,testdata,type = "raw")
pred_log
testdata$class
con_matrix_log <- table(pred_log,testdata$class,dnn = c("Prediction","Actual"))
con_matrix_log
confusionMatrix(con_matrix_log)


# naive Bayes classifier

modelnb <- train(class~.,methods="nb",data = traindata,trControl = control)
pred_nb <- predict(modelnb,testdata,type = "raw")
pred_nb
testdata$class
con_matrix_nb <- table(pred_nb,testdata$class,dnn = c("Prediction","Actual"))
con_matrix_nb
confusionMatrix(con_matrix_nb)


# decision tree

modeldt <- train(class~.,methods="c5.0",data = traindata,trControl = control)
pred_dt <- predict(modeldt,testdata,type = "raw")
pred_dt
testdata$class
con_matrix_dt <- table(pred_dt,testdata$class,dnn = c("Prediction","Actual"))
con_matrix_dt
confusionMatrix(con_matrix_dt)

# random forest

modelrf <- train(class~.,methods="rf",data = traindata,trControl = control)
pred_rf <- predict(modelrf,testdata,type = "raw")
pred_rf
testdata$class
con_matrix_rf <- table(pred_dt,testdata$class,dnn = c("Prediction","Actual"))
con_matrix_rf
confusionMatrix(con_matrix_rf)


# support vector machine - radial kernel

modelsvmr <- train(class~.,methods="svmRadial",data = traindata,trControl = control)
pred_svmr <- predict(modelsvmr,testdata,type = "raw")
pred_svmr
testdata$class
con_matrix_svmr <- table(pred_svmr,testdata$class,dnn = c("Prediction","Actual"))
con_matrix_svmr
confusionMatrix(con_matrix_svmr)


# support vector machine- polynomial kernel

modelsvmp <- train(class~.,methods="svmPoly",data = traindata,trControl = control)
pred_svmp <- predict(modelsvmr,testdata,type = "raw")
pred_svmp
testdata$class
con_matrix_svmp <- table(pred_svmp,testdata$class,dnn = c("Prediction","Actual"))
con_matrix_svmp
confusionMatrix(con_matrix_svmp)



#  ROC-Curve
roc<-evalm(list(modellog,modelnb,modeldt,modelrf,modelsvmr,modelsvmp),gnames = c("LOG","NB","DT","RF","SVM-RADIAL","SVM-POLY"))


saveRDS(modelsvmr,"modelsvmr.rds")
