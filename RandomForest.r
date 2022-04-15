set.seed(41)
library(ROCR)
library(randomForest)
library(caret)
library(InformationValue)

load("train_data")
load("test_data")
load("train_smote")

nominal_class_metrics <- function(predicted, actual) {
    actual <- as.numeric(levels(actual))[actual] # nolint

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
    # plot(roc_perf2, avg = "threshold") # nolint

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
train$Class <- as.factor(train$Class)
test$Class <- as.factor(test$Class)
train_smote$class <- as.factor(train_smote$class)

######################## TRAINING #######################

fraud_rf <- randomForest(Class ~ ., data = train, ntree = 10, importance = TRUE)
rf_smote <- randomForest(class ~ ., data = train_smote, ntree = 10, importance = TRUE) # nolint
saveRDS(rf_smote, "rf_smote.rds")
saveRDS(fraud_rf, "randomForest.rds")
fraud_pred <- predict(fraud_rf, train, type = "prob")[, 2]
auc_roc_metric(fraud_pred, train$Class, 0.5)

oob <- trainControl(method = "oob")
rf_grid <- expand.grid(mtry = 1:30)
fraud_rf_tune <- train(Class ~., data = train, method = "rf", 
    trControl = oob, verbose = TRUE, tuneGrid = rf_grid)
fraud_rf_tune$bestTune

auc_roc_metric(predict(fraud_rf_tune, train, type = "prob")[, 2], train$Class, 0.5) # nolint

######################## TESTING ##########################
fraud_pred_test <- predict(fraud_rf, newdata = test, type = "prob")[, 2]
rf_smote_test <- predict(rf_smote, newdata = test, type = "prob") [, 2]
cutoff1 <- optimalCutoff(test$Class, fraud_pred_test)
cutoff <- optimalCutoff(test$Class, rf_smote_test)
auc_roc_metric(fraud_pred_test, test$Class, cutoff1)
auc_roc_metric(rf_smote_test, test$Class, cutoff)

fraud_pred_test2 <- predict(fraud_rf_tune, newdata = test, type = "prob")[, 2]
cutoff2 <- optimalCutoff(test$Class, fraud_pred_test2)
auc_roc_metric(fraud_pred_test2, test$Class, cutoff2)
