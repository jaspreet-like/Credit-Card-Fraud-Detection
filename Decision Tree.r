library(ISLR)
library(tree)
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
train$Class <- as.factor(train$Class)
test$Class <- as.factor(test$Class)

######################## TRAINING #######################
fraud_tree <- tree(Class ~ ., data = train)
summary(fraud_tree)
plot(fraud_tree)
text(fraud_tree, pretty = 0)

######################## TESTING ##########################
fraud_pred <- predict(fraud_tree, newdata = test, type = "class")

table(fraud_pred, test$Class)
nominal_class_metrics(fraud_pred, test$Class)
