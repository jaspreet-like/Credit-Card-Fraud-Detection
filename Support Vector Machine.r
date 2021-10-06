library(e1071)
load("train_data")
load("test_data")

set.seed(41)
t <- train[sample(nrow(train), 10000), ]
str(t)
model1 <- svm(Class ~ ., data = t, kernel = "linear", cost = 10, scale = FALSE)
model1_pred <- predict(model1, type = "response")

##################### Try ####################
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
y <- c(3, 4, 5, 4, 8, 10, 10, 11, 14, 20, 23, 24, 32, 34, 35, 37, 42, 48, 53, 60) # nolint
t <- data.frame(x, y)
plot(t, pch = 16)
library(e1071)
model_svm <- svm(y ~ x, t)
pred <- predict(model_svm, t)
points(t$x, pred, col = "blue", pch = 4)
