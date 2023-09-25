# --1-- Get data from table
df <- read.csv('C:\\Users\\Arseny\\Desktop\\Programs\\Code\\R\\Lab_2\\Risk.csv')
# --2-- View first six rows from table
print("Visualize started data from table: ") 
head(df)
# --3-- Load library for using palette abilities and make color palette
library(RColorBrewer)
palette <- brewer.pal(3, "Set2")
# --4-- Remake nominal categorical values like factor
df[, 'Risk'] <- factor(df[, 'Risk'])
df[, 'Gender'] <- factor(df[, 'Gender'])
print("Remake symbol data to factor values")
head(df)
# --5-- Visualize palette (matrix of scatter plots variables to each other)
plot(x=df, col = palette[factor(df[, 'Risk'])], pch = 19)
# --6-- Make scatter plot for Age from BMI
plot(x=df$BMI,y=df$Age,
     col = palette[factor(df[, 'Risk'])], 
     pch = 19,
     xlab = 'BMI',
     ylab = 'Age',
     main = 'Age-BMI scatter'
)
# --7-- create random state variable for repeating results
set.seed(42)
# --8-- make list of splited data for test and train with some frequency
idx <- createDataPartition(y=df$Risk,p=0.8,list = FALSE)
# --9-- Creating train and test data frames
train <- df[idx,]
test <- df[-idx,]
# --10-- Print number of objects in train and test
cat("Number of train objects:", nrow(train), "and test:", nrow(test))
# --11-- Import package for classification and regression training
library(caret)
# --12-- Train k nearest neighbours method
# set up the training control with cross-validation
ctrl <- trainControl(method = "cv", number = 10)
# set up the tuning grid for the k parameter
kGrid <- expand.grid(k = 1:20)
# train the knn3 model with cross-validation
cat("Training process of Neural Network:")
knn_model <- train(Risk ~ Age + BMI + Gender + State.Rate, 
                   data = train, 
                   method = "knn", 
                   trControl = ctrl, 
                   tuneGrid = kGrid)
# get best k value
best_k <- knn_model$bestTune$k
cat("Best k value:", best_k, "\n")
# --13-- Predict results for test data
knn_pred <- predict(knn_model, newdata = test,type = 'raw')
cat("Prediction results on test data:")
table(x = knn_pred,y = test$Risk)
# --14-- Print some types of accuracy for knn
knn_cm <- confusionMatrix(knn_pred,test$Risk)
knn_cm
# --15-- Load library for decision tree classifier
library(tree)
# --16-- train model
tree_model <- tree(formula = Risk ~ Age + BMI + Gender + State.Rate, data = train)
# --17-18-- Create and visualize current tree structure
summary(tree_model)
plot(tree_model)
text(tree_model, font=3)
# --19-- Predict test values with this model
tree_pred <- predict(tree_model,newdata = test,type='class')
# --20-- Write confussion matrix
cat("Data of confusion matrix: ")
table(x=tree_pred,y=test$Risk)
# --21-- Write accuracy results
cat("Different type info and accuracy of tree model")
tree_cm <- confusionMatrix(tree_pred,test$Risk)
tree_cm
# --22-- Import package for working with neural networks
library(nnet)
# --23-- Train neural network with given parameters
nn_model <- nnet(formula=Risk~Age+BMI+Gender+State.Rate, data=train,
                 size=10, decay=0.0001, maxit=500)
# --24-26-- Import package to visualizing neural network structure and apply it
library(NeuralNetTools)
plotnet(nn_model)
# --27-28-- Predict test data with NN model and plot confusion matrix
nn_pred <- predict(nn_model,newdata = test, type='class')
# --29-- Print different accuracy results
nn_cm <- confusionMatrix(factor(nn_pred), test$Risk)
nn_cm
# --30-- Compare all the model accuracy and define what is better for this task
print(knn_cm$overall[1])
print(tree_cm$overall[1])
print(nn_cm$overall[1])
best_value = max(c(knn_cm$overall[1], tree_cm$overall[1], nn_cm$overall[1]))
best_index = which.max(c(knn_cm$overall[1], tree_cm$overall[1], nn_cm$overall[1]))
cat("The best algorithm for this task is:", c('knn', 'tree', 'nn')[best_index],
    "with value:", best_value)
