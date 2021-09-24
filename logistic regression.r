######################### Loading the dataset ########################
library(ROCR)
library(caret)
library(ggplot2)

# Loaded with variable name "train"
load("train_data")

# Loaded with variable name "test"
load("test_data")
# Variable names can also be verified using ls() function in console

# Function to measure nominal class performance of the model
nominal_class_metrics <- function(predicted, actual) {
    TP <- sum(actual[predicted])
    FP <- sum(!actual[predicted])
    TN <- sum(!actual[!predicted])
    FN <- sum(actual[!predicted])

    print(paste("True Negatives:", TN, "False Positives:", FP))
    print(paste("False Negatives:", FN, "True Positives:", TP))
    print("--------")
    print(paste("Accuracy of the model is:", (TP + TN) / (TP + FP + TN + FN)))

    prec <- (TP) / (TP + FP)
    print(paste("Precision of the model is:", prec))

    recall <- (TP) / (TP + FN)
    print(paste("Recall/Sensitivity of the model is:", recall))

    print(paste("Specificity of the model is:", (TN) / (TN + FP)))

    print(paste("F1 score of the model is:", (2 * prec * recall) / (prec + recall))) # nolint

    print(paste("G score of the model is:", sqrt(prec * recall())))

    POS <- TP + FN
    NEG <- FP + TN
    PPOS <- TP + FP
    PNEG <- FN + TN
    mcc <- (TP * TN + FP * FN) / sqrt(POS * NEG * PPOS * PNEG)
    print(paste("Mathews Correlation Coefficient is:", mcc))

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

################# Fitting a Logistic Regression Model ###################

# Fitted response to binomial family(BUT WHY?)
model1 <- glm(Class ~ ., data = train, family = binomial)
summary(model1)

# Another model ignoring the non-useful features
model2 <- glm(Class ~ V1 + V4 + V6 + V8 + V10 + V13 + V14 + V16 + V20 + V21 + V22 + V23 + V27 + Amount, data = train, family = binomial) # nolint
summary(model2)

# This is a vector storing probabilities of each observation to be fraudulent
model1_prob <- predict(model1, type = "response")
model2_prob <- predict(model2, type = "response")

# I tried finding a package which can tell me all these metrics scores
# Found one - 'caret'. I tried a lot to implement it, but couldn't.
# So, I wrote my own function which calculates all these metric scores 
table(train$Class, model1_prob > 0.5)
nominal_class_metrics((model1_prob > 0.5), train$Class)

table(train$Class, model2_prob > 0.5)
nominal_class_metrics((model2_prob > 0.5), train$Class)

# Need a lot of experimenting with this ROCR package, will commit the final last part later # nolint
roc_pred <- prediction(model1_prob, train$Class)
str(roc_pred)
roc_perf <- performance(roc_pred, "tpr", "fpr")
str(roc_perf)
plot(roc_perf, colorize = TRUE, text.adj = c(-0.2, 1.7))
