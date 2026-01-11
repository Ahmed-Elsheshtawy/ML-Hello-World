import random

training_inputs = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
correct_outputs = [(4*x)+5 for x in training_inputs]

testing_inputs = [3, 4, 5]
correct_test_outputs = [(4*x)+5 for x in testing_inputs]

class SimpleModel:
    def __init__(self):
        self.weight = random.randrange(-20, 20)
        self.bias = random.randrange(-20, 20)
        self.learning_rate = 0.01
        self.predicted_outputs = []
    
    def feedforward(self, x):
        return (self.weight * x) + self.bias # Output of the model
    
    def calculate_loss(self, predicted, correct):
        return (predicted - correct) ** 2 # Mean Squared Error
    
    def calculate_weight_gradient(self, x, predicted, correct):
        return 2 * x * (predicted - correct) # Gradient with respect to weight
    
    def calculate_bias_gradient(self, predicted, correct):
        return 2 * (predicted - correct) # Gradient with respect to bias

    def backpropagate(self, x, predicted, correct):
        weight_gradient = self.calculate_weight_gradient(x, predicted, correct)
        bias_gradient = self.calculate_bias_gradient(predicted, correct)
        
        self.weight -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

    def train(self, training_inputs, correct_outputs, epochs):
        for epoch in range(epochs):
            loss = []
            for x, correct in zip(training_inputs, correct_outputs):
                # Calculate loss and print it
                loss.append(self.calculate_loss(self.feedforward(x), correct))

                # Perform feedforward and backpropagation
                predicted = self.feedforward(x)
                self.backpropagate(x, predicted, correct)
            
            total_loss = sum(loss) / len(loss)
            print(f"Epoch {epoch}, Loss: {total_loss}")
        
        print(f"Trained weight: {self.weight}, Trained bias: {self.bias}")


# Create and train the model
model = SimpleModel()
EPOCHS = 100
model.train(training_inputs, correct_outputs, epochs=EPOCHS)

print("\nTesting the trained model:")
# Yay! The code is complete. You can now test the model with testing inputs.
for x, correct in zip(testing_inputs, correct_test_outputs):
    predicted = model.feedforward(x)
    print(f"Input: {x}, Predicted: {predicted}, Correct: {correct}")