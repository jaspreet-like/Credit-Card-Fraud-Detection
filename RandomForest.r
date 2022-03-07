set.seed(41)
library(ROCR)
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
auc_roc_metric <- function(model_prob, actual) {
    actual_numeric <- as.numeric(actual)
    roc_pred <- prediction(model_prob, actual_numeric)
    roc_perf <- performance(roc_pred, "rec", "prec")
    plot(roc_perf, avg = "threshold")

    roc_auc <- performance(roc_pred, "auc")
    area <- roc_auc@y.values[[1]]
    print(paste("Area under ROC curve: ", round(area, digits = 4)))
}
train$Class <- as.factor(train$Class)
test$Class <- as.factor(test$Class)

######################## TRAINING #######################

fraud_rf <- randomForest(Class ~ ., data = train, ntree = 10, importance = TRUE)

fraud_pred <- predict(fraud_rf, newdata = train)
nominal_class_metrics(fraud_pred, train$Class)

oob <- trainControl(method = "oob")
rf_grid <- expand.grid(mtry = 1:30)
fraud_rf_tune <- train(Class ~., data = train, method = "rf", 
    trControl = oob, verbose = TRUE, tuneGrid = rf_grid)

nominal_class_metrics(predict(fraud_rf_tune, train), train$Class)
fraud_rf_tune$bestTune

######################## TESTING ##########################
fraud_pred_test <- predict(fraud_rf, newdata = test)
nominal_class_metrics(fraud_pred_test, test$Class)

fraud_pred_test2 <- predict(fraud_rf_tune, newdata = test)
nominal_class_metrics(fraud_pred_test2, test$Class)
