library(ranger)
library(caret)
library(data.table)
credit_card_data <- read.csv("C:/users/akhil/Desktop/creditcard.csv")
dim(credit_card_data)
head(credit_card_data,6)
#Data Scaling: It ensures that there are no extreme values in the dataset, so that the model is interfered by it. It is also called feature extraction.

credit_card_data$Amount = scale(credit_card_data$Amount)

NewData = credit_card_data[,-c(1)]

head(NewData)

#Data Modeling: Diving the dataset into 80% of training set and 20% of test data. 

library(caTools)
set.seed(123)

data_sample = sample.split(NewData$Class,SplitRatio = 0.80)

train_data = subset(NewData,data_sample == TRUE)
test_data = subset(NewData, data_sample = FALSE)
dim(train_data)
dim(test_data)

Logistic_Model = glm(Class~.,test_data,family = binomial())

summary(Logistic_Model)


#Fitting Logistic Regression Model: Used to determine the probability of an outcome such as fraud/not fraud.

Logistic_Model = glm(Class~.,test_data, family = binomial())

#Visualising the model.

plot(Logistic_Model)

library(pROC)

lr.predict <- predict(Logistic_Model,train_data,probability = TRUE)
auc.gbm = roc(test_data$Class,lr.predict,plot = TRUE, col = 'blue')

#Fitting the Decision Tree: Plot the outcomes of a decision.
library(rpart)
library(rpart.plot)
decisionTree_model <- rpart(Class~., credit_card_data, method = 'class')
predicted_val <- predict(decisionTree_model, credit_card_data, type = 'class')
probability <- predict(decisionTree_model, credit_card_data, type = 'prob')
rpart.plot(decisionTree_model)


#Artificial Neural Network 

library(neuralnet)
ANN_model = neuralnet(Class ~., train_data,linear.output = FALSE)
plot(ANN_model)

predANN = compute(ANN_model, test_data)
resultANN = predANN$net.result
resultANN = ifelse(resultANN > 0.5, 1.0)

#Gradient Boosting (GBM)
library(gbm, quietly = TRUE)
system.time(
  model_gbm <- gbm(Class~., distribution =  "bernoulli"
                  , data = rbind(train_data, test_data)
                  , n.trees = 500
                  , interaction.depth = 3
                  , n.minobsinnode = 100
                  , shrinkage = 0.01
                  , bag.fraction = 0.5
                  , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
                  
  )
  )
gbm.iter = gbm.perf(model_gbm, method = "test")





model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)
plot(model_gbm)

gbm_test =  predict(model_gbm, newdata = test_data, n.trees = gbm.iter)

gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, col = "red")

print(gbm_auc)
  