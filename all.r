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
    # predicted <- as.logical(as.integer(levels(predicted)[predicted]))

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
auc_roc_metric <- function(model_prob, actual_factor, CutOff) { # nolint
    actual_numeric <-  as.numeric(levels(actual_factor))[actual_factor] # nolint
    roc_pred <- prediction(model_prob, actual_numeric)

    # Precision-Recall Curve
    roc_perf <- performance(roc_pred, "prec", "rec")
    plot(roc_perf, avg = "threshold")

    # ROC Curve
    # roc_perf2 <- performance(roc_pred, "tpr", "fpr") # nolint

    # Here, just add a parameter named -- "colorize - TRUE, add = TRUE"
    # plot(roc_perf2, avg = "threshold", colorize = TRUE) # nolint

    a <- nominal_class_metrics(model_prob > 0.5, actual_factor)
    print(paste("-----------------X----------------"))
    b <- nominal_class_metrics(model_prob > CutOff, actual_factor)

    # Marking these precision recall on the precison-recall curve
    points(a[[2]], a[[1]], pch = 20, cex = 3, col = "red")
    points(b[[2]], b[[1]], pch = 20, cex = 3, col = "forest green")

    roc_auc <- performance(roc_pred, "auc")
    area <- roc_auc@y.values[[1]]
    print(paste("-----------------X----------------"))
    print(paste("Area under ROC curve: ", round(area, digits = 4)))
}

######################## TRAINING #######################

lr <- readRDS("logisticRegression.rds")
svm <- readRDS("svm.rds")
dt <- readRDS("decisionTree.rds")
rf <- readRDS("randomForest.rds")

######################## TESTING ##########################

lr_pred <- predict(lr, newdata = test, type = "response")
foo <- predict(svm, newdata = test,
                        decision.values = TRUE, probability = TRUE)
svm_pred <- attr(foo, "probabilities") [, 2]
dt_pred <- predict(dt, newdata = test, type = "prob")[, 2]
rf_pred <- predict(rf, newdata = test, type = "prob")[, 2]

lr_cutoff <- optimalCutoff(test$Class, lr_pred) # nolint
svm_cutoff <- optimalCutoff(test$Class, svm_pred)
dt_cutoff <- optimalCutoff(test$Class, dt_pred)
rf_cutoff <- optimalCutoff(test$Class, rf_pred)


test$Class <- as.factor(test$Class)
auc_roc_metric(lr_pred, test$Class, lr_cutoff)
auc_roc_metric(svm_pred, test$Class, svm_cutoff)
auc_roc_metric(dt_pred, test$Class, dt_cutoff)
auc_roc_metric(rf_pred, test$Class, rf_cutoff)
