library(ISLR)
library(tree)
library(ROCR)
library(rpart)
library(readr)
library(caTools)
library(dplyr)
library(party)
library(rpart.plot)

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

# Using 'tree' package

fraud_tree1 <- tree(Class ~ ., data = train)
summary(fraud_tree1)
plot(fraud_tree1)
text(fraud_tree1, pretty = 0)
title(main = "Unpruned Decision tree")
# To get a vector of probabilities, we can use type="vector"
fraud_pred1 <- predict(fraud_tree1, newdata = train, type = "class")
nominal_class_metrics(fraud_pred1, train$Class)
fraud_pred_test1 <- predict(fraud_tree1, newdata = test, type = "class")
nominal_class_metrics(fraud_pred_test1, test$Class)
set.seed(41)
cv_fraud <- cv.tree(fraud_tree1, FUN = prune.misclass)
min_idx <- which.min(cv_fraud$dev)
min_size <- cv_fraud$size[min_idx]
misclassification_rate <- cv_fraud$dev / nrow(train)
plot(cv_fraud$size, cv_fraud$dev / nrow(train), type = "b",
     xlab = "Tree Size", ylab = "CV Misclassification Rate")
fraud_prune <- prune.misclass(fraud_tree1, best = min_size)
summary(fraud_prune)
plot(fraud_prune)
text(fraud_prune, pretty = 0)
title(main = "Pruned Decision tree")
fraud_pred2 <- predict(fraud_prune, newdata = train, type = "class")
nominal_class_metrics(fraud_pred2, train$Class)
fraud_pred_test2 <- predict(fraud_prune, newdata = test, type = "class")
nominal_class_metrics(fraud_pred_test2, test$Class)



# Using 'rpart' package

fraud_tree2 <- rpart(Class ~ ., data = train)
plotcp(fraud_tree2)
fraud_pred3 <- predict(fraud_tree2, train, type = "class")
nominal_class_metrics(fraud_pred3, train$Class)
fraud_pred_test3 <- predict(fraud_tree2, test, type = "class")
nominal_class_metrics(fraud_pred_test3, test$Class)

min_cp <- fraud_tree2$cptable[which.min(fraud_tree2$cptable[, "xerror"]), "CP"]
fraud_prune2 <- prune(fraud_tree2, cp = min_cp)
rpart.plot(fraud_prune2)
prp(fraud_prune2, faclen = 0, cex = 0.8, extra = 1)
fraud_pred4 <- predict(fraud_prune2, train, type = "class")
nominal_class_metrics(fraud_pred4, train$Class)
fraud_pred_test4 <- predict(fraud_prune2, test, type = "class")
nominal_class_metrics(fraud_pred_test4, test$Class)



######################## TESTING ##########################
fraud_pred <- predict(fraud_tree, newdata = test, type = "class")
nominal_class_metrics(fraud_pred, test$Class)
