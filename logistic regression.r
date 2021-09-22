train <- read.csv("train.csv")
test <- read.csv("test.csv")

n <- nrow(train)
m <- nrow(test)

frauds_train <- 0
for (i in 1:n) {
    frauds_train <- frauds_train + train$Class[i]
}
frauds_test <- 0
for (i in 1:m) {
    frauds_test <- frauds_test + test$Class[i]
}
print(paste("Training:", frauds_train, "out of", n))
print(paste("Testing:", frauds_test, "out of", m))

head(train)
summary(train)
y <- train$Class
x <- as.matrix(train[, 1:30])
dim(x)

model <- glm(Class ~ . - Time, data = train, family = binomial)
summary(model)
predict <- predict(model, type = "response")

library(ROCR)
roc_pred <- prediction(predict, CC$Class)
