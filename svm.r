library(kernlab)
library(fifer)
library(caret)

input = read.csv("HR_comma_sep.csv")

cost_matrix = matrix(c(0, 50, 2, -5),
                    byrow = TRUE,
                    nrow = 2,
                    ncol = 2)

set.seed(10)
training_data = stratified(df = input, group = "left", size = 0.7)
testing_data = input[!duplicated(rbind(training_data, input))[-seq_len(nrow(training_data))], ]

X_train = subset(training_data, select = -left)
Y_train = training_data[,'left']
X_train[,'sales'] = as.numeric(X_train[,'sales'])
X_train[,'salary'] = as.numeric(X_train[,'salary'])

X_test = subset(testing_data, select = -left)
Y_test = testing_data[,'left']
X_test[,'sales'] = as.numeric(X_test[,'sales'])
X_test[,'salary'] = as.numeric(X_test[,'salary'])

# linear model
linear_model = ksvm(x = as.matrix(X_train), y = as.factor(Y_train), kernel = "vanilladot", type = "C-svc")
predictions_linear = predict(linear_model, X_test)
mat_linear = confusionMatrix(predictions_linear, Y_test)
## for quadratic model
polynomial_model = ksvm(x = as.matrix(X_train), y = as.factor(Y_train), kernel = "polydot", type = "C-svc")
predictions_polynomial = predict(polynomial_model, X_test)
mat_polynomial = confusionMatrix(predictions_polynomial, Y_test)
## for Gaussian model
gaussian_model = ksvm(x = as.matrix(X_train), y = as.factor(Y_train), kernel = "rbfdot", type = "C-svc")
predictions_gaussian = predict(gaussian_model, X_test)
mat_gaussian = confusionMatrix(predictions_gaussian, Y_test)
## for tanh sigmoid model
sigmoid_model = ksvm(x = as.matrix(X_train), y = as.factor(Y_train), kernel = "tanhdot", type = "C-svc")
predictions_sigmoid = predict(sigmoid_model, X_test)
mat_sigmoid= confusionMatrix(predictions_sigmoid, Y_test)

# Results of Linear model
## Accuracy
linear_accuracy = mat_linear$overall["Accuracy"]
linear_accuracy = unname(linear_accuracy)
linear_accuracy
## Precision
linear_precision = mat_linear$table[1,1] / (mat_linear$table[1,1] + mat_linear$table[1,2])
linear_precision
## Recall
linear_recall = mat_linear$byClass["Sensitivity"]
linear_recall = unname(linear_recall)
linear_recall
## F_measure
linear_f_measure = 2 * linear_recall * linear_precision / (linear_recall + linear_precision)
linear_f_measure
## Cost
linear_cost = mat_linear$table[1,1] * cost_matrix[1,1] + mat_linear$table[1,2] * cost_matrix[1,2] + mat_linear$table[2,1] * cost_matrix[2,1] + mat_linear$table[2,2] * cost_matrix[2,2]
linear_cost

# Results of Polynomial model
## Accuracy
polynomial_accuracy = mat_polynomial$overall["Accuracy"]
polynomial_accuracy = unname(polynomial_accuracy)
polynomial_accuracy
## Precision
polynomial_precision = mat_polynomial$table[1,1] / (mat_polynomial$table[1,1] + mat_polynomial$table[1,2])
polynomial_precision
## Recall
polynomial_recall = mat_polynomial$byClass["Sensitivity"]
polynomial_recall = unname(polynomial_recall)
polynomial_recall
## F_measure
polynomial_f_measure = 2 * polynomial_recall * polynomial_precision / (polynomial_recall + polynomial_precision)
polynomial_f_measure
## Cost
polynomial_cost = mat_polynomial$table[1,1] * cost_matrix[1,1] + mat_polynomial$table[1,2] * cost_matrix[1,2] + mat_polynomial$table[2,1] * cost_matrix[2,1] + mat_polynomial$table[2,2] * cost_matrix[2,2]
polynomial_cost

# Results of Gaussian model
## Accuracy
gaussian_accuracy = mat_gaussian$overall["Accuracy"]
gaussian_accuracy = unname(gaussian_accuracy)
gaussian_accuracy
## Precision
gaussian_precision = mat_gaussian$table[1,1] / (mat_gaussian$table[1,1] + mat_gaussian$table[1,2])
gaussian_precision
## Recall
gaussian_recall = mat_gaussian$byClass["Sensitivity"]
gaussian_recall = unname(gaussian_recall)
gaussian_recall
## F_measure
gaussian_f_measure = 2 * gaussian_recall * gaussian_precision / (gaussian_recall + gaussian_precision)
gaussian_f_measure
## Cost
gaussian_cost = mat_gaussian$table[1,1] * cost_matrix[1,1] + mat_gaussian$table[1,2] * cost_matrix[1,2] + mat_gaussian$table[2,1] * cost_matrix[2,1] + mat_gaussian$table[2,2] * cost_matrix[2,2]
gaussian_cost

# Results of tanh Sigmoid model
## Accuracy
sigmoid_accuracy = mat_sigmoid$overall["Accuracy"]
sigmoid_accuracy = unname(sigmoid_accuracy)
sigmoid_accuracy
## Precision
sigmoid_precision = mat_sigmoid$table[1,1] / (mat_sigmoid$table[1,1] + mat_sigmoid$table[1,2])
sigmoid_precision
## Recall
sigmoid_recall = mat_sigmoid$byClass["Sensitivity"]
sigmoid_recall = unname(sigmoid_recall)
sigmoid_recall
## F_measure
sigmoid_f_measure = 2 * sigmoid_recall * sigmoid_precision / (sigmoid_recall + sigmoid_precision)
sigmoid_f_measure
## Cost
sigmoid_cost = mat_sigmoid$table[1,1] * cost_matrix[1,1] + mat_sigmoid$table[1,2] * cost_matrix[1,2] + mat_sigmoid$table[2,1] * cost_matrix[2,1] + mat_sigmoid$table[2,2] * cost_matrix[2,2]
sigmoid_cost

