# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Jeeva K

### Register Number: 212223230090

```python

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


torch.manual_seed(71)
X=torch.linspace(1,50,50).reshape(-1,1)
e=torch.randint(-8,9,(50,1),dtype=torch.float)
y=2*X+1+e


plt.scatter(X,y,color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()

class Model(nn.Module):
  def __init__(self,in_features,out_features):
    super().__init__()
    self.linear=nn.Linear(in_features,out_features)

  def forward(self,x):
    return self.linear(x)
torch.manual_seed(59)
model=Model(1,1)
initial_weight=model.linear.weight.item()
initial_bias=model.linear.bias.item()

print(f"Initial Weight: {initial_weight:.8f},Initial Bias: {initial_bias:.8f}\n")


loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
epochs=100
losses=[]
for epoch in range(1,epochs+1):
  optimizer.zero_grad()
  y_pred=model(X)
  loss=loss_function(y_pred,y)
  losses.append(loss.item())
  loss.backward()
  optimizer.step()
  print(f"epoch: {epoch:2} loss: {loss.item():10.8f}"
         f"weight: {model.linear.weight.item():10.8f}"
         f"bias: {model.linear.bias.item():10.8f}")


plt.plot(range(epochs),losses,color="Blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()


final_weight=model.linear.weight.item()
final_bias=model.linear.bias.item()
print("\nName : Jeeva K")
print("Register No : 212223230090")
print(f"\nFinal Weight : {final_weight:.8f}, Final Bias : {final_bias:.8f}")


x1=torch.tensor([X.min().item(),X.max().item()])
y1=x1*final_weight+final_bias

plt.scatter(X,y,label="Original Data")
plt.plot(x1,y1,'r',label="Best=Fit line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()


x_new=torch.tensor([[120.0]])
y_new_pred=model(x_new).item()

print(f"\nPrediction for x = 120 : {y_new_pred:.8f}")



```



## OUTPUT

### Dataset Information
<img width="627" height="456" alt="image" src="https://github.com/user-attachments/assets/2b42971f-5b8a-4aac-b634-c1c8e0b549fa" />

### Initial weight & Bias:
<img width="228" height="42" alt="image" src="https://github.com/user-attachments/assets/5bad038a-f4d1-4a89-bb5e-954729bb04ad" />

### Training Loss Vs Iteration Plot:
<img width="549" height="753" alt="image" src="https://github.com/user-attachments/assets/3cf68ace-5fad-438a-8814-e8fd5be4b5dc" />

<img width="550" height="747" alt="image" src="https://github.com/user-attachments/assets/8ff4e9d0-ebd4-4552-952e-b16405a15a2b" />


<img width="466" height="207" alt="image" src="https://github.com/user-attachments/assets/eae7a144-2c70-4a84-b087-46dc027bfcdb" />

### Loss Curve:
<img width="649" height="462" alt="image" src="https://github.com/user-attachments/assets/faaca89b-fea7-4041-b6a1-f0a37c7b7d8f" />

### Final weight & Bias:
<img width="405" height="83" alt="Screenshot 2025-08-26 104954" src="https://github.com/user-attachments/assets/62617c07-d3b1-440f-ac4f-a933d5c05fc0" />

### Best Fit line plot:
<img width="645" height="452" alt="image" src="https://github.com/user-attachments/assets/bb8e12e6-13b1-43f7-b4a9-c66b7e20c7ab" />


### New Sample Data Prediction:
<img width="321" height="46" alt="image" src="https://github.com/user-attachments/assets/5ca93812-d601-41f1-bc69-d8528e547344" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
