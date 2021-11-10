######################### Loading the dataset ########################

# Loaded with variable name "train"
load("train_data")

# Loaded with variable name "test"
load("test_data")
# Variable names can also be verified using ls() function in console

# Function to measure nominal class performance of the model
# NOTE: predicted should be as.logical and actual should be as.numeric
nominal_class_metrics <- function(predicted, actual) {
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



################## Getting familiar with the dataset ###################

# very handy command to get a one liner brief summary of features
str(train)
str(test)

# Just checking how many missing values are there in the dataset
sum(is.na(train))
sum(is.na(test))

# just checking how imbalanced is our dataset
print(paste("Training frauds:", sum(train$Class), "out of", nrow(train)))
print(paste("Testing frauds:", sum(test$Class), "out of", nrow(test)))




################# Training ###################

# Fitted response to binomial family as our response is only 0 and 1
model1 <- glm(Class ~ ., data = train, family = binomial)
summary(model1)

# New model ignoring the non-useful features
model2 <- glm(Class ~ V1 + V4 + V6 + V8 + V10 + V13 + V14 + V16 + V20 + V21 + V22 + V23 + V27 + Amount, data = train, family = binomial) # nolint
summary(model2)

# This is a vector storing probabilities of each observation to be fraudulent
model2_prob <- predict(model2, type = "response")

# Confusion matrix, assuming CutOff to be 0.5
table(train$Class, model2_prob > 0.5)

# I wrote my own function which calculates all the relevant performance metric scores # nolint
nominal_class_metrics((model2_prob > 0.5), train$Class)

# ROC Curve and AUC (area under the curve)
library(ROCR)
roc_pred <- prediction(model2_prob, train$Class)
roc_perf <- performance(roc_pred, "tpr", "fpr")
plot(roc_perf, avg = "threshold")

roc_auc <- performance(roc_pred, "auc")
area <- roc_auc@y.values[[1]]
print(paste("Area under ROC curve: ", round(area, digits = 4)))

# Precision-Recall Curve to visualize the model better
roc_perf1 <- performance(roc_pred, "rec", "prec")
plot(roc_perf1, avg = "threshold")

# For calculating the optimal cutoff probability for our probability model
library(InformationValue)
CutOff <- optimalCutoff(train$Class, model2_prob) # nolint
CutOff

# To make sense of this cutoff, we look at highest 500 probabilities
head(sort(model2_prob, decreasing = TRUE), 500)
sum(model2_prob > 0.5)
sum(model2_prob > 0.12)

# Comparing the performance metrics wrt optimal cutoff-0.12 and our initial assumed cutoff-0.5 # nolint
nominal_class_metrics(model2_prob > CutOff, train$Class)
nominal_class_metrics(model2_prob > 0.5, train$Class)

# Marking these precision recall on the precison-recall curve
points(0.8415, 0.8, pch = 20, cex = 3, col = "red")
points(0.9065, 0.6904, pch = 20, cex = 3, col = "forest green")



######################## TESTING ##############################

# Making the probability vector as per our model on the test dataset
test_model <- predict(model2, newdata = test, type = "response")

# AUC for test dataset and precision-recall curve
roc_pred2 <- prediction(test_model, test$Class)

roc_auc <- performance(roc_pred2, "auc")
area <- roc_auc@y.values[[1]]
print(paste("Area under ROC curve: ", round(area, digits = 4)))

roc_perf2 <- performance(roc_pred2, "rec", "prec")
plot(roc_perf2, avg = "threshold")

# Calculating Optimal cutoff for test data
CutOff <- optimalCutoff(test$Class, test_model) # nolint

# Performance of our model on test dataset # nolint 
nominal_class_metrics(test_model > CutOff, test$Class)
nominal_class_metrics(test_model > 0.5, test$Class)

# Marking these precision recall points on the precison-recall curve
points(0.727, 0.693, pch = 20, cex = 3, col = "red")
points(0.761, 0.653, pch = 20, cex = 3, col = "forest green")
