######################### Preparing the dataset ########################
set.seed(41)

# Loaded with variable name "train"
load("train_data")

# Loaded with variable name "test"
load("test_data")

# Performance mertics function
# NOTE: Both predicted and actual should be as.factor
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

# We create a "small" dataset having "fr" = number of fraudulent observations
# and "any" = number of random samples from original train dataset
fr <- 100
any <- 10000
t <- train[train$Class == 1, ]
t <- t[sample(nrow(t), fr), ]
small <- rbind(t, train[sample(nrow(train), any), ])

fr <- 33
any <- 3333
t <- test[test$Class == 1, ]
t <- t[sample(nrow(t), fr), ]
small_test <- rbind(t, test[sample(nrow(test), any), ])

# We make "Class" feature of all datasets as factor
train$Class <- as.factor(train$Class) # nolint
test$Class <- as.factor(test$Class) # nolint
small$Class <- as.factor(small$Class) # nolint
small_test$Class <- as.factor(small_test$Class) # nolint

################# Fitting SVM Model on small training dataset ###################
library(e1071)

model11 <- svm(Class ~ ., data = small, kernel = "polynomial", scale = TRUE)
model21 <- svm(Class ~ ., data = small, kernel = "polynomial", cost = 100, scale = TRUE) # nolint
model31 <- svm(Class ~ ., data = small, kernel = "polynomial", coef = 100, scale = TRUE) # nolint
model41 <- svm(Class ~ ., data = small, kernel = "polynomial", cost = 10, coef = 10, scale = TRUE) # nolint
model51 <- svm(Class ~ ., data = small, kernel = "radial", scale = TRUE)
model61 <- svm(Class ~ ., data = small, kernel = "sigmoid", scale = TRUE)

pred11 <- predict(model11, type = "response")
pred21 <- predict(model21, type = "response")
pred31 <- predict(model31, type = "response")
pred41 <- predict(model41, type = "response")
pred51 <- predict(model51, type = "response")
pred61 <- predict(model61, type = "response")

nominal_class_metrics(pred11, small$Class)
nominal_class_metrics(pred21, small$Class)
nominal_class_metrics(pred31, small$Class)
nominal_class_metrics(pred41, small$Class)
nominal_class_metrics(pred51, small$Class)
nominal_class_metrics(pred61, small$Class)

pred11 <- predict(model11, newdata = small_test, type = "response")
pred21 <- predict(model21, newdata = small_test, type = "response")
pred31 <- predict(model31, newdata = small_test, type = "response")
pred41 <- predict(model41, newdata = small_test, type = "response")
pred51 <- predict(model51, newdata = small_test, type = "response")
pred61 <- predict(model61, newdata = small_test, type = "response")

nominal_class_metrics(pred11, small_test$Class)
nominal_class_metrics(pred21, small_test$Class)
nominal_class_metrics(pred31, small_test$Class)
nominal_class_metrics(pred41, small_test$Class)
nominal_class_metrics(pred51, small_test$Class)
nominal_class_metrics(pred61, small_test$Class)

######################### TRAINING ############################

# NOTE: It'll take around 5 minutes to model svm on train dataset
svm_model <- svm(Class ~ ., data = train, kernel = "polynomial", scale = TRUE)
svm_pred <- predict(svm_model, type = "response")

# Performance of our model on training data
nominal_class_metrics(svm_pred, train$Class)

######################## TESTING ##############################

# Making the probability vector as per our model on the test dataset
test_model <- predict(svm_model, newdata = test, type = "response")

# Performance of our model on test dataset
nominal_class_metrics(test_model, test$Class)
