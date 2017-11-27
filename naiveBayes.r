library("caret")
library("caTools")
library("splitstackshape")

data = read.csv("HR_comma_sep.csv")
# Convert left to a factor
data$left <- as.factor(data$left)

set.seed(100)
#sample <- sample.split(data$left, SplitRatio = .8)
train.index <- createDataPartition(data$left, p = .8, list = FALSE)

# train <- subset(data, sample==TRUE)
train <- data[train.index,]
test <- data[-train.index,]
# test <- subset(data, sample==FALSE)

train_control<- trainControl(method="cv", number = 10)

nb_model <- train(left~., trControl = train_control, data=train, method="nb")

predictions<- predict(nb_model,test)

nb_modelbinded <- cbind(test,predictions)

confusionMatrix<- confusionMatrix(nb_modelbinded$left, nb_modelbinded$predictions)

confusionMatrix

