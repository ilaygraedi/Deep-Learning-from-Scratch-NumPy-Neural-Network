import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class layer:
    def __init__(self,inputs,neurons):
        self.weights = 0.01 * np.random.randn(inputs,neurons) 
        self.bias = np.zeros((1,neurons))
        self.momentum_w = np.zeros_like(self.weights)
        self.momentum_b = np.zeros_like(self.bias)
        self.cache_w = np.zeros_like(self.weights)
        self.cache_b = np.zeros_like(self.bias)
    
    def forward(self,input):
        self.input = input
        self.output = np.matmul(input, self.weights) + self.bias
    
    def backward(self,gradient):
        self.dweights = np.matmul(self.input.T, gradient)
        self.dbias = np.sum(gradient, keepdims=True, axis=0)
        self.dinput = np.matmul(gradient, self.weights.T)

class ReLU:
    def forward(self,input):
        self.input = input
        self.output = np.maximum(0,input)
   
    def backward(self,gradient):
        self.dinput = gradient.copy()
        self.dinput[self.input <= 0] = 0

class loss_softmax_categoricalcrossentropy:
    def forward_softmax(self,input):
        self.input = input
        self.output = np.exp(input - np.max(input,axis=1,keepdims=True))
        self.output = self.output/np.sum(self.output,axis=1,keepdims=True)
    
    def forward_crossentropy(self,target,predicted):
        self.target =target
        self.predicted = predicted
        clip_predicted =np.clip(predicted, 1e-7, 1 - 1e-7)
        x = -np.log(clip_predicted)
        self.loss = x * target
        self.loss = np.sum(self.loss,axis=1)
        self.loss = np.mean(self.loss)
    
    def backward(self):
        sampels = len(self.predicted)
        self.dinput = self.predicted - self.target
        self.dinput /= sampels

class optimizer_ADAM:
    def __init__(self,learning_rate = 0.001, momentum=0.9,beta2 = 0.999,epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta2 =beta2
        self.epsilon = epsilon
        self.iteration = 0
    
    def update(self,layer:layer):
        
        layer.momentum_w = self.momentum * layer.momentum_w + (1-self.momentum) * layer.dweights
        layer.cache_w = self.beta2 * layer.cache_w + (1-self.beta2) * layer.dweights**2
        m_hat = layer.momentum_w/(1-self.momentum**self.iteration)
        cache_mhat = layer.cache_w/(1-self.beta2**self.iteration)
        layer.weights = layer.weights - self.learning_rate * m_hat/(np.sqrt(cache_mhat)+self.epsilon)

        layer.momentum_b = self.momentum * layer.momentum_b + (1-self.momentum) * layer.dbias
        layer.cache_b = self.beta2 * layer.cache_b + (1-self.beta2) * layer.dbias**2
        b_hat = layer.momentum_b/(1-self.momentum**self.iteration)
        cache_bhat = layer.cache_b/(1-self.beta2**self.iteration)
        layer.bias = layer.bias - self.learning_rate * b_hat/(np.sqrt(cache_bhat)+self.epsilon)
    
    def iteration_update(self):
        self.iteration += 1

table =pd.read_csv("ML/train.csv")
table = table.to_numpy()
target = table[:,0]
one_hot_encoding = np.zeros((len(target),10))
rows = np.arange(0,len(target))
one_hot_encoding[rows,target] =1 
data = table[:,1:]
data = data/255

layer1 = layer(784,128)
layer2 = layer(128,10)
relu = ReLU()
loss = loss_softmax_categoricalcrossentropy()
optimizer = optimizer_ADAM()
batch_size = 100

accuracy_history = []
loss_history = []
fig,( ax1, ax2) = plt.subplots(1,2)

for epoch in range(30):
    for i in range(0,len(data),batch_size):
        batch_data = data[i:i+batch_size]
        batch_onehot = one_hot_encoding[i:i+batch_size]
        layer1.forward(batch_data)
        relu.forward(layer1.output)
        layer2.forward(relu.output)
        loss.forward_softmax(layer2.output)
        loss.forward_crossentropy(batch_onehot,loss.output)
        loss.backward()
        layer2.backward(loss.dinput)
        optimizer.iteration_update()
        optimizer.update(layer2)
        relu.backward(layer2.dinput)
        layer1.backward(relu.dinput)
        optimizer.update(layer1)
    
    layer1.forward(data)
    relu.forward(layer1.output)
    layer2.forward(relu.output)
    loss.forward_softmax(layer2.output)
    loss.forward_crossentropy(one_hot_encoding,loss.output)
    accuracy =  np.argmax(loss.output,axis=1)
    real = np.argmax(one_hot_encoding,axis=1)
    accuracy_history.append(np.mean(accuracy==real))
    loss_history.append(loss.loss)
    
    print(f"loss: {loss_history[-1]} accuracy: {accuracy_history[-1]}")
    
# test =pd.read_csv("ML/test.csv")
# test = test.to_numpy()
# test = test/255
# layer1.forward(test)
# relu.forward(layer1.output)
# layer2.forward(relu.output)
# loss.forward_softmax(layer2.output)
# prediction = np.argmax(loss.output,axis=1)
# imgid = np.arange(1,28001)
# dic = {"ImageId":imgid,"Label":prediction}
# train = pd.DataFrame(dic)
# train.to_csv("submission.csv",index=False)

ax1.plot(accuracy_history, color = "red")
ax2.plot(loss_history, color = "blue")
ax1.set_title("Accuracy")
ax2.set_title("Loss")
ax1.set_xlabel("Epoch")
ax2.set_xlabel("Epoch")

plt.show()

