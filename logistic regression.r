######################### Loading the dataset ########################
library(ROCR)
library(caret)
library(ggplot2)

# Loaded with variable name "train"
load("train_data")

# Loaded with variable name "test"
load("test_data")
# Variable names can also be verified using ls() function in console

# Function to measure nominal class performance of the model
nominal_class_metrics <- function(predicted, actual) {
    TP <- sum(actual[predicted])
    FP <- sum(!actual[predicted])
    TN <- sum(!actual[!predicted])
    FN <- sum(actual[!predicted])

    print(paste("True Negatives:", TN, "False Positives:", FP))
    print(paste("False Negatives:", FN, "True Positives:", TP))
    print("--------")
    print(paste("Accuracy of the model is:", (TP + TN) / (TP + FP + TN + FN)))

    prec <- (TP) / (TP + FP)
    print(paste("Precision of the model is:", prec))

    recall <- (TP) / (TP + FN)
    print(paste("Recall/Sensitivity of the model is:", recall))

    print(paste("Specificity of the model is:", (TN) / (TN + FP)))

    print(paste("F1 score of the model is:", (2 * prec * recall) / (prec + recall))) # nolint

    print(paste("G score of the model is:", sqrt(prec * recall())))

    POS <- TP + FN
    NEG <- FP + TN
    PPOS <- TP + FP
    PNEG <- FN + TN
    mcc <- (TP * TN + FP * FN) / sqrt(POS * NEG * PPOS * PNEG)
    print(paste("Mathews Correlation Coefficient is:", mcc))

}



################## Getting familiar with the dataset ###################

# very handy command to get a one liner brief summary of features
str(train)
str(test)

# Just checking how many missing values are there in the dataset
sum(is.na(train))
sum(is.na(test))

# just checking how imbalanced is our dataset
print(paste("Training frauds:", sum(train$Class), "out of", nrow(train)))
print(paste("Testing frauds:", sum(test$Class), "out of", nrow(test)))

################# Fitting a Logistic Regression Model ###################

# Fitted response to binomial family(BUT WHY?)
model1 <- glm(Class ~ ., data = train, family = binomial)
summary(model1)

# Another model ignoring the non-useful features
model2 <- glm(Class ~ V1 + V4 + V6 + V8 + V10 + V13 + V14 + V16 + V20 + V21 + V22 + V23 + V27 + Amount, data = train, family = binomial) # nolint
summary(model2)

# This is a vector storing probabilities of each observation to be fraudulent
model1_prob <- predict(model1, type = "response")
model2_prob <- predict(model2, type = "response")

# I tried finding a package which can tell me all these metrics scores
# Found one - 'caret'. I tried a lot to implement it, but couldn't.
# So, I wrote my own function which calculates all these metric scores 
table(train$Class, model1_prob > 0.5)
nominal_class_metrics((model1_prob > 0.5), train$Class)

# Metric score on train dataset, assuming CutOff to be 0.5
table(train$Class, model2_prob > 0.5)
nominal_class_metrics((model2_prob > 0.5), train$Class)

# ROC Curve to get area under the curve
roc_pred <- prediction(model2_prob, train$Class)
roc_perf <- performance(roc_pred, "tpr", "fpr")
roc_auc <- performance(roc_pred,"auc")
area <- roc_auc@y.values[[1]]
area
plot(roc_perf, avg="threshold", spread.estimate="boxplot" )

# For calculating the optimal cutoff probability for our probability model
library(InformationValue)
CutOff <- optimalCutoff(train$Class,model2_prob)

# Checking the optimal cutoff performance on the original dataset
nominal_class_metrics(model2_prob>CutOff,train$Class)

######################## TESTING ##############################
# Making the probability vector as per our model on the test dataset
test_model <- predict(model2, newdata=test, type="response")

# Performance of our model on the test dataset
nominal_class_metrics(test_model>CutOff,test$Class)
# Comparing it with default 0.5 cutoff
nominal_class_metrics(test_model>0.5,test$Class)

# AUC for test dataset
roc_pred <- prediction(test_model, test$Class)
roc_perf <- performance(roc_pred, "tpr", "fpr")
roc_auc <- performance(roc_pred,"auc")
area <- roc_auc@y.values[[1]]
area
plot(roc_perf, avg="threshold", spread.estimate="boxplot" )
