######################### Preparing the dataset ########################
set.seed(41)
library(e1071)
library(ROCR)
library(InformationValue)

# Loaded with variable name "train"
load("train_data")

# Loaded with variable name "test"
load("test_data")

# Performance metrics function
# NOTE: Both predicted and actual should be as.factor
nominal_class_metrics <- function(predicted, actual) {
    actual <- as.numeric(levels(actual))[actual] # nolint
    predicted <- as.logical(as.integer(levels(predicted)[predicted]))

    TP <- sum(actual[predicted])
    FP <- sum(!actual[predicted])
    TN <- sum(!actual[!predicted])
    FN <- sum(actual[!predicted])

    print(paste("True Negatives:", TN, "False Positives:", FP))
    print(paste("False Negatives:", FN, "True Positives:", TP))
    print("--------")
    acc <- (TP + TN) / (TP + FP + TN + FN)
    print(paste("Accuracy of the model is:", round(acc, digits = 4))) # nolint

    prec <- (TP) / (TP + FP)
    print(paste("Precision of the model is:", round(prec, digits = 4)))

    recall <- (TP) / (TP + FN)
    print(paste("Recall/Sensitivity of the model is:", round(recall, digits = 4))) # nolint

    spec <- (TN) / (TN + FP)
    print(paste("Specificity of the model is:", round(spec, digits = 4))) # nolint

    f1 <- (2 * prec * recall) / (prec + recall)
    print(paste("F1 score of the model is:", round(f1, digits = 4))) # nolint

    g <- sqrt(prec * recall)
    print(paste("G score of the model is:", round(g, digits = 4))) # nolint

    POS <- TP + FN
    NEG <- FP + TN
    PPOS <- TP + FP
    PNEG <- FN + TN
    mcc <- (TP * TN + FP * FN) / (sqrt(POS * NEG) * sqrt(PPOS * PNEG))
    print(paste("Mathews Correlation Coefficient is:", round(mcc, digits = 4)))
}

# We make "Class" feature of all datasets as factor
train$Class <- as.factor(train$Class) # nolint
test$Class <- as.factor(test$Class) # nolint

######################### TRAINING ############################

# Caution: It'll take around 5 minutes to model each svm on train dataset
svm_model1 <- svm(Class ~ ., data = train, kernel = "polynomial", scale = TRUE)
svm_model2 <- svm(Class ~ ., data = train, kernel = "radial", scale = TRUE)
svm_model3 <- svm(Class ~ ., data = train, kernel = "sigmoid", scale = TRUE)
svm_pred1 <- predict(svm_model1, type = "response")
svm_pred2 <- predict(svm_model2, type = "response")
svm_pred3 <- predict(svm_model3, type = "response")

# Performance of our model on training data
nominal_class_metrics(svm_pred1, train$Class)
nominal_class_metrics(svm_pred2, train$Class)
nominal_class_metrics(svm_pred3, train$Class)


######################## TESTING ##############################

# Making the probability vector as per our model on the test dataset
test_pred1 <- predict(svm_model1, newdata = test, type = "response")
test_pred2 <- predict(svm_model2, newdata = test, type = "response")
test_pred3 <- predict(svm_model3, newdata = test, type = "response")

# Performance of our model on test data
nominal_class_metrics(test_pred1, test$Class)
nominal_class_metrics(test_pred2, test$Class)
nominal_class_metrics(test_pred3, test$Class)