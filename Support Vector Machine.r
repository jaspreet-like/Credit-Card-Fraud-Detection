######################### Preparing the dataset ########################
set.seed(41)
library(e1071)
library(ROCR)
library(InformationValue)

# Loaded with variable name "train"
load("train_data")

# Loaded with variable name "test"
load("test_data")

# Small dataset with 1010 observations for quick check test
a <- train[1:1000, ]
b <- train[which(train$Class == 1), ]
c <- b[1:10, ]
small_train <- rbind(a, c)

# Performance metrics function
# NOTE: Predicted should be as.logical and actual should be as.factor
nominal_class_metrics <- function(predicted, actual) {
    actual <- as.numeric(levels(actual))[actual] # nolint
    # predicted <- as.logical(as.integer(levels(predicted)[predicted])) # nolint

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

# NOTE: model_prob should be as.numeric and actual_factor should be as.factor
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
# We make "Class" feature of all datasets as factor
train$Class <- as.factor(train$Class) # nolint
test$Class <- as.factor(test$Class) # nolint
small_train$Class <- as.factor(small_train$Class) # nolint

######################### TRAINING ############################

# Caution: It'll take around 10 sec to train on "small_train" dataset and
# about 8-10 minutes on "train" dataset
svm_model1 <- svm(Class ~ ., data = train, probability = TRUE)
saveRDS(svm_model1, "svm.rds")
 svm_model1 <- readRDS("svm.rds")

svm_pred1 <- predict(svm_model1, train, probability = TRUE)
model <- attr(svm_pred1, "probabilities") [, 2]
CutOff1 <- optimalCutoff(train$Class, model) # nolint

# Metrics
auc_roc_metric(model, train$Class, 0.5)

######################## TESTING ##############################

# Making the probability vector as per our model on the test dataset
test_pred1 <- predict(svm_model1, newdata = test,
                        decision.values = TRUE, probability = TRUE)
test_model <- attr(test_pred1, "probabilities") [, 2]
CutOff2 <- optimalCutoff(test$Class, test_model) # nolint

# Reason why 'CutOff' is so low
head(sort(test_model, decreasing = TRUE), 500)

# Metrics
auc_roc_metric(test_model, test$Class, CutOff2)
