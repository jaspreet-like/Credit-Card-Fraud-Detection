######################## Handling Unbalanced Data ##########################

set.seed(41)
library(smotefamily)
library(e1071)
library(ROCR)
library(InformationValue)
library(ISLR)
library(tree)
library(rpart)
library(readr)
library(caTools)
library(dplyr)
library(party)
library(rpart.plot)
library(randomForest)
library(caret)
library(grid)
library(gridExtra)

load("train_data")
load("test_data")
train$Class <- as.factor(train$Class) # nolint
test$Class <- as.factor(test$Class) # nolint
# smote <- SMOTE(train[,  -31], train[, 31], K = 5) # nolint
# train_smote <- smote$data # nolint
# train_smote$class <- as.integer(train_smote$class) # nolint
# save(train_smote, file = "train_smote") # nolint
load("train_smote")
train_smote$class <- as.factor(train_smote$class)

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

    x <- c(TN, FP, FN, TP, acc, prec, recall, spec, f1, g, mcc)
    return(x)
}
auc_roc_metric <- function(model_prob, actual_factor, CutOff, m) { # nolint
    actual_numeric <-  as.numeric(levels(actual_factor))[actual_factor] # nolint
    roc_pred <- prediction(model_prob, actual_numeric)

    # Precision-Recall Curve
    roc_perf <- performance(roc_pred, "prec", "rec")
    plot(roc_perf, avg = "threshold", xlim = c(0, 1), ylim = c(0, 1)) # nolint
    if (m == 1) title(main = "Logistic Regression")
    if (m == 2) title(main = "SVM", cex.main = 1)
    if (m == 3) title(main = "Decision Tree", cex.main = 1)
    if (m == 4) title(main = "Random Forest", cex.main = 1)

    a <- nominal_class_metrics(model_prob > 0.5, actual_factor)
    print(paste("-----------------X----------------"))
    b <- nominal_class_metrics(model_prob > CutOff, actual_factor)

    # Marking these precision recall on the precison-recall curve
    points(a[7], a[6], pch = 20, cex = 3, col = "red")
    points(b[7], b[6], pch = 20, cex = 3, col = "forest green")

    roc_auc <- performance(roc_pred, "auc")
    area <- roc_auc@y.values[[1]]
    print(paste("-----------------X----------------"))
    print(paste("Area under ROC curve: ", round(area, digits = 4)))

    b <- append(b, area)
    return(b)
}
roc_curve <- function(model_prob, actual_factor, CutOff, m){ # nolint
    actual_numeric <-  as.numeric(levels(actual_factor))[actual_factor] # nolint
    roc_pred <- prediction(model_prob, actual_numeric)
    roc_perf2 <- performance(roc_pred, "tpr", "fpr")
    if (m == 1) plot(roc_perf2, avg = "threshold", col = "black")
    if (m == 2) plot(roc_perf2, avg = "threshold", col = "blue", add = TRUE)
    if (m == 3) plot(roc_perf2, avg = "threshold", col = "orange", add = TRUE) # nolint
    if (m == 4) plot(roc_perf2, avg = "threshold", col = "pink", add = TRUE)
}


####################### TRAINING ###################################33

lr_smote <- readRDS("lr_smote.rds")
svm_smote <- readRDS("svm_smote.rds")
dt_smote <- readRDS("dt_smote.rds")
rf_smote <- readRDS("rf_smote.rds")

######################## TESTING ##########################

lrb_pred <- predict(lr_smote, newdata = test, type = "response")
foo <- predict(svm_smote, newdata = test,
                decision.values = TRUE, probability = TRUE)
svmb_pred <- attr(foo, "probabilities") [, 2]
dtb_pred <- predict(dt_smote, newdata = test, type = "prob")[, 2]
rfb_pred <- predict(rf_smote, newdata = test, type = "prob") [, 2]

lrb_cutoff <- optimalCutoff(test$Class, lrb_pred, optimiseFor = "Both")
svmb_cutoff <- optimalCutoff(test$Class, svmb_pred, optimiseFor = "Both")
dtb_cutoff <- optimalCutoff(test$Class, dtb_pred, optimiseFor = "Both")
rfb_cutoff <- optimalCutoff(test$Class, rfb_pred, optimiseFor = "Both")


#pdf("Prec-Recall.pdf", width = 10, height = 10) # nolint
png("smote_prec_recall_curves.png", width = 1024, height = 1024, units = "px")
par(mfrow = c(2, 2))
l <- auc_roc_metric(lrb_pred, test$Class, lrb_cutoff, 1)
s <- auc_roc_metric(svmb_pred, test$Class, svmb_cutoff, 2)
d <- auc_roc_metric(dtb_pred, test$Class, dtb_cutoff, 3)
r <- auc_roc_metric(rfb_pred, test$Class, rfb_cutoff, 4)
#mtext("Prec-Recall Curves for Original Data", outer = TRUE,  cex = 1.5, line = -1.4, col = "red") # nolint
dev.off()

#pdf("roc.pdf", width = 4, height = 4)
png("smote_roc_curves.png", width = 300, height = 300, units = "px")
roc_curve(lrb_pred, test$Class, lrb_cutoff, 1)
roc_curve(svmb_pred, test$Class, svmb_cutoff, 2)
roc_curve(dtb_pred, test$Class, dtb_cutoff, 3)
roc_curve(rfb_pred, test$Class, rfb_cutoff, 4)
#mtext("ROC Curves for Original Data", outer = TRUE,  cex = 1, line = -1.4, col = "red") # nolint
legend(x = "bottomright", legend = c("Logistic Regression", "SVM",
    "Decision Tree", "Random Forest"), fill = c("black", "blue",
        "orange", "pink"), cex = 0.8)
dev.off()

pn <- rbind(l[1:4], s[1:4], d[1:4], r[1:4])
colnames(pn) <- c("True Negatives", "False Positives",
                    "False Negatives", "True Positives")
rownames(pn) <- c("Logistic Regression", "SVM",
                    "Decision Tree", "Random Forest")
pn <- as.table(pn)

metrics <- rbind(l[5:12], s[5:12], d[5:12], r[5:12])
metrics <- format(round(metrics, 4))
colnames(metrics) <- c("Accuracy", "Precision", "Recall/Sensitivity",
                         "Specificity", "F1 Score", "G score", "MCC", "AUC-ROC")
rownames(metrics) <- c("Logistic Regression", "SVM",
                    "Decision Tree", "Random Forest")
metrics <- as.table(metrics)
mytable1 <- tableGrob(pn, theme = ttheme_default
            (core = list(bg_params = list(fill = "grey99"))))
mytable2 <- tableGrob(metrics, theme = ttheme_default
            (core = list(bg_params = list(fill = "grey99"))))
#pdf("metrics.pdf", width = 10, height = 8)
#grid.arrange(mytable1, mytable2, ncol = 1, nrow = 2)
png("smote_metrics_table1.png", width = 560, height = 240, units = "px")
grid.draw(mytable1)
dev.off()
png("smote_metrics_table2.png", width = 760, height = 240, units = "px")
grid.draw(mytable2)
dev.off()
