######################### LOADING DATASET ########################
library(ROCR)
library(InformationValue)

# Loaded with variable name "train"
load("train_data")

# Loaded with variable name "test"
load("test_data")
# Variable names can also be verified using ls() function in console

# Function to measure nominal class performance of the model
# NOTE: predicted should be as.logical and actual should be as.factor
nominal_class_metrics <- function(predicted, actual_factor) {
    actual <- as.numeric(levels(actual_factor))[actual_factor] # nolint

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

# NOTE: model_prob should be as.numeric and actual can be factor
auc_roc_metric <- function(model_prob, actual_factor, CutOff) {
    actual_numeric <-  as.numeric(levels(actual_factor))[actual_factor] # nolint
    roc_pred <- prediction(model_prob, actual_numeric)

    # Precision-Recall Curve
    roc_perf <- performance(roc_pred, "prec", "rec")
    plot(roc_perf, avg = "threshold")

    # ROC Curve
    # roc_perf2 <- performance(roc_pred, "tpr", "fpr") # nolint
    # plot(roc_perf2, avg = "threshold") # nolint

    print(paste("Metrics with 0.5 as cutoff"))
    a <- nominal_class_metrics(model_prob > 0.5, actual_factor)
    print(paste("                           "))
    print(paste("Metrics with optimal cutoff"))
    b <- nominal_class_metrics(model_prob > CutOff, actual_factor)

    # Marking these precision recall on the precison-recall curve
    points(a[[2]], a[[1]], pch = 20, cex = 3, col = "red")
    points(b[[2]], b[[1]], pch = 20, cex = 3, col = "forest green")

    roc_auc <- performance(roc_pred, "auc")
    area <- roc_auc@y.values[[1]]
    print(paste("-----------------X----------------"))
    print(paste("Area under ROC curve: ", round(area, digits = 4)))
}


################## GETTING FAMILIAR WITH DATASET ###################

# very handy command to get a one liner brief summary of features
str(train)
str(test)

# Just checking how many missing values are there in the dataset
sum(is.na(train))
sum(is.na(test))

# Just checking how imbalanced is our dataset
print(paste("Training frauds:", sum(train$Class), "out of", nrow(train)))
print(paste("Testing frauds:", sum(test$Class), "out of", nrow(test)))




################# TRAINING ###################

# Fitted response to binomial family as our response is only 0 and 1
model1 <- glm(Class ~ ., data = train, family = binomial)
summary(model1)

# New model ignoring the non-useful features
model2 <- glm(Class ~ V1 + V4 + V6 + V8 + V10 + V13 + V14 + V16 + V20 + V21 + V22 + V23 + V27 + Amount, data = train, family = binomial) # nolint
saveRDS(model2, "logisticRegression.rds")
# model2 <- readRDS("logisticRegression.rds") # nolint
summary(model2)

# This is a vector storing probabilities of each observation to be fraudulent
model2_prob <- predict(model2, type = "response")

# For calculating the optimal cutoff probability for our probability model
CutOff <- optimalCutoff(train$Class, model2_prob) # nolint

# I wrote my own metric functions which calculates all the relevant performance metric scores # nolint
train$Class <- as.factor(train$Class)
auc_roc_metric(model2_prob, train$Class, CutOff)

# Why is this CutOff so small? - we look at highest 500 probabilities
head(sort(model2_prob, decreasing = TRUE), 500)
sum(model2_prob > 0.5)
sum(model2_prob > 0.12)



######################## TESTING ##############################

# Making the probability vector as per our model on the test dataset
test_model <- predict(model2, newdata = test, type = "response")

# Calculating Optimal cutoff for test data
CutOff <- optimalCutoff(test$Class, test_model) # nolint
CutOff

# AUC for test dataset and precision-recall curve
test$Class <- as.factor(test$Class)
auc_roc_metric(test_model, test$Class, CutOff)
