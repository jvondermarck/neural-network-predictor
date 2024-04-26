import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# The model is our neural network
model = Sequential()

# We need to add layers to our model
# Input layer:
# Dense means that all the neurons are connected to each other
# We will give 3 neurons to the input layer
# We will give a 1D array as input (not a matrix)
model.add(Dense(units=3, input_shape=[1]))

# Hidden layers:
# Layers that will process and combine the input data to be able to predict the output
# We will give 64 neurons to the hidden layer (not a big number otherwise it will take a lot of time to train the model)
model.add(Dense(units=64))

# Output layer:
# The output layer will have only 1 neuron because we want to predict only 1 value
model.add(Dense(units=1))


# Declaration of the input and output :
input_data = np.array([1, 2, 3, 4, 5])
output_data = np.array([2, 4, 6, 8, 10]) # The output data is the double of the input data. If the IA is working well, it will predict the double of the input data

# Compilation of the model and give the function to correct and optimize itself
# mean_squared_error: we will consider that a value is well predicted when the square of the difference between the predicted value and the real value is 0
# adam: the optimizer that will correct the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
# epochs: the number of times the model will train itself
model.fit(input_data, output_data, epochs=1000) 

# Testing the model
while True:
    x = float(input("Enter a number to predict: "))
    
    prediction = model.predict(np.array([x]))[0][0]  # Extract the predicted value from the array
    real_value = x * 2
    difference = real_value - prediction
    
    print(' - Prediction from the Neural Network: {:.2f}'.format(prediction))
    print(' - Real value: {:.2f}'.format(real_value))
    print(' - Difference: {:.2f}'.format(difference))