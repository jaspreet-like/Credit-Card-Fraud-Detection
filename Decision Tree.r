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

######################## TRAINING #######################

# Using 'rpart' package

fraud_tree2 <- rpart(Class ~ ., data = train, method = "class")
plotcp(fraud_tree2)

min_cp <- fraud_tree2$cptable[which.min(fraud_tree2$cptable[, "xerror"]), "CP"]
fraud_prune2 <- prune(fraud_tree2, cp = min_cp)
saveRDS(fraud_prune2, "decisionTree.rds")
rpart.plot(fraud_prune2)
prp(fraud_prune2, faclen = 0, cex = 0.8, extra = 1)

fraud_pred3 <- predict(fraud_prune2, train, type = "prob")[, 2]
cutoff1 <- optimalCutoff(train$Class, fraud_pred3)
auc_roc_metric(fraud_pred3, train$Class, cutoff1)

fraud_pred_test3 <- predict(fraud_prune2, test, type = "prob")[, 2]
cutoff2 <- optimalCutoff(test$Class, fraud_pred_test3)
auc_roc_metric(fraud_pred_test3, test$Class, cutoff2)


# Using 'tree' package

fraud_tree1 <- tree(Class ~ ., data = train, method = "class")
summary(fraud_tree1)
pdf("tree.pdf", width = 6, height = 4)
plot(fraud_tree1)
text(fraud_tree1, pretty = 0)
title(main = "Unpruned Decision tree")
dev.off()
# To get a vector of probabilities, we can use type="vector"
fraud_pred1 <- predict(fraud_tree1, type = "vector")[, 2]
auc_roc_metric(fraud_pred1, train$Class, 0.5)

fraud_pred_test1 <- predict(fraud_tree1, newdata = test, type = "vector")[, 2]
auc_roc_metric(fraud_pred_test1, test$Class, 0.5)
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
fraud_pred2 <- predict(fraud_prune, newdata = train, type = "vector")[, 2]
auc_roc_metric(fraud_pred2, train$Class, 0.5)
fraud_pred_test2 <- predict(fraud_prune, newdata = test, type = "vector")[, 2]
auc_roc_metric(fraud_pred_test2, test$Class, 0.5)
