set.seed(41)
library(e1071)
library(ROCR)
library(InformationValue)
library(ISLR)
library(tree)
library(rpart)
library(readr)
library(caTools)
library(dplyr)
library(party)
library(rpart.plot)
library(randomForest)
library(caret)

load("train_data")
load("test_data")

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
auc_roc_metrics <- function(model_prob, actual) {
    actual_numeric <- as.numeric(actual)
    roc_pred <- prediction(model_prob, actual_numeric)
    roc_perf <- performance(roc_pred, "rec", "prec")
    plot(roc_perf, avg = "threshold")

    roc_auc <- performance(roc_pred, "auc")
    area <- roc_auc@y.values[[1]]
    print(paste("Area under ROC curve: ", round(area, digits = 4)))
}
nominal_class_metric <- function(predicted, actual) {
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

    return(list(prec, recall))
}

# NOTE: model_prob should be as.numeric and actual can be integer/numeric
auc_roc_metric <- function(model_prob, actual, CutOff) {
    actual_numeric <- as.numeric(actual)
    roc_pred <- prediction(model_prob, actual_numeric)

    # Precision-Recall Curve
    roc_perf <- performance(roc_pred, "prec", "rec")
    plot(roc_perf, avg = "threshold")

    # ROC Curve
    # roc_perf2 <- performance(roc_pred, "tpr", "fpr")
    # plot(roc_perf2, avg = "threshold")

    a <- nominal_class_metrics(model_prob > 0.5, actual)
    print(paste("-----------------X----------------"))
    b <- nominal_class_metrics(model_prob > CutOff, actual)

    # Marking these precision recall on the precison-recall curve
    points(a[[2]], a[[1]], pch = 20, cex = 3, col = "red")
    points(b[[2]], b[[1]], pch = 20, cex = 3, col = "forest green")

    roc_auc <- performance(roc_pred, "auc")
    area <- roc_auc@y.values[[1]]
    print(paste("-----------------X----------------"))
    print(paste("Area under ROC curve: ", round(area, digits = 4)))
}

model2 <- glm(Class ~ V1 + V4 + V6 + V8 + V10 + V13 + V14 + V16 + V20 + V21 + V22 + V23 + V27 + Amount, data = train, family = binomial) # nolint
test_model <- predict(model2, newdata = test, type = "response")
CutOff <- optimalCutoff(test$Class, test_model) # nolint
auc_roc_metric(test_model, test$Class, CutOff)

train$Class <- as.factor(train$Class)
test$Class <- as.factor(test$Class)

svm_model1 <- svm(Class ~ ., data = train, kernel = "polynomial", scale = TRUE)
test_pred1 <- predict(svm_model1, newdata = test, type = "response")
nominal_class_metrics(test_pred1, test$Class)