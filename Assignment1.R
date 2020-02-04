library(keras)
imdb <- dataset_imdb(num_words = 10000)# top 10,000 most frequently occurring words in the training data
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb
# Preparing Data
vectorize_sequences <- function(sequences, dimension = 10000) {
  # Create an all-zero matrix of shape (len(sequences), dimension)
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1
  results
}
# Our vectorized training data
x_train <- vectorize_sequences(train_data)
# Our vectorized test data
x_test <- vectorize_sequences(test_data)
str(x_train[1,])
# Our vectorized labels
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)
val_indices <- 1:10000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]
###################################################################
#************************************** Direct***************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "tanh",input_shape = c(10000)) %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
model %>% evaluate(x_test, y_test)
#********************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "tanh" , input_shape = c(10000)) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
model %>% evaluate(x_test, y_test)
#**********************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "tanh", input_shape = c(10000)) %>% 
  layer_dense(units = 8, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
model %>% evaluate(x_test, y_test)
#*************************************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "tanh", input_shape = c(10000)) %>% 
  layer_dense(units = 32, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
str(history)
plot(history)
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#************************************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "tanh",input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "tanh") %>%
  layer_dense(units = 8, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
model %>% evaluate(x_test, y_test)
#*****************************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "tanh",input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "tanh") %>%
  layer_dense(units = 8, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
str(history)
plot(history)
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results



###################################################### regularizer_l2 ##################################################################################
#64**************************************************************************************************************************

model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "tanh",kernel_regularizer  = regularizer_l2(0.001) ,input_shape = c(10000)) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
 model %>% evaluate(x_test, y_test)

#********************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "tanh",kernel_regularizer  = regularizer_l2(0.001) , input_shape = c(10000)) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))

model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#**********************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "tanh",kernel_regularizer  = regularizer_l2(0.001) , input_shape = c(10000)) %>% 
  layer_dense(units = 8, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
str(history)
plot(history)
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#*************************************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "tanh", kernel_regularizer  = regularizer_l2(0.001) ,input_shape = c(10000)) %>% 
  layer_dense(units = 32, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
str(history)
plot(history)
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#************************************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "tanh", kernel_regularizer  = regularizer_l2(0.001) ,input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "tanh") %>%
  layer_dense(units = 8, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
str(history)
plot(history)
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#*****************************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "tanh", kernel_regularizer  = regularizer_l2(0.001) ,input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "tanh") %>%
  layer_dense(units = 8, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
str(history)
plot(history)
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
###################################################### Dropout ##################################################################################
#**************************************************************************************************************************

model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "tanh",kernel_regularizer  = regularizer_l2(0.001) ,input_shape = c(10000)) %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
model %>% evaluate(x_test, y_test)
#********************************************************************************************************
###REg 2

model <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "tanh",kernel_regularizer  = regularizer_l1(0.001) ,input_shape = c(10000)) %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
model %>% evaluate(x_test, y_test)
#************************************




model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "tanh",kernel_regularizer  = regularizer_l2(0.001) , input_shape = c(10000)) %>% 
  layer_output(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
str(history)
plot(history)
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#**********************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "tanh",kernel_regularizer  = regularizer_l2(0.001) , input_shape = c(10000)) %>% 
  layer_output(rate = 0.5) %>%
  layer_dense(units = 16, activation = "tanh") %>%
  layer_output(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
str(history)
plot(history)
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#*************************************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "tanh", kernel_regularizer  = regularizer_l2(0.001) ,input_shape = c(10000)) %>% 
  layer_output(rate = 0.5) %>%
  layer_dense(units = 8, activation = "tanh") %>%
  layer_output(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
str(history)
plot(history)
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#************************************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "tanh", kernel_regularizer  = regularizer_l2(0.001) ,input_shape = c(10000)) %>% 
  layer_output(rate = 0.5) %>%
  layer_dense(units = 8, activation = "tanh") %>%
  layer_output(rate = 0.5) %>%
  layer_dense(units = 4, activation = "tanh") %>%
  layer_output(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
str(history)
plot(history)
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#****************************************************************************************************************
model <- keras_model_sequential() %>% 
  layer_dense(units = 8, activation = "tanh", kernel_regularizer  = regularizer_l2(0.001) ,input_shape = c(10000)) %>% 
  layer_output(rate = 0.5) %>%
  layer_dense(units = 4, activation = "tanh") %>%
  layer_output(rate = 0.5) %>%
  layer_dense(units = 2, activation = "tanh") %>%
  layer_output(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
str(history)
plot(history)
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results
#####************

model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "tanh",kernel_regularizer  = regularizer_l2(0.001) ,input_shape = c(10000)) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val))
model %>% fit(x_train, y_train, epochs = 2, batch_size = 512)
model %>% evaluate(x_test, y_test)