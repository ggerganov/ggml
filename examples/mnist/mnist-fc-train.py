import gguf
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from time import time

input_size = 784 # img_size = (28,28) ---> 28*28=784 in total
hidden_size = 500 # number of nodes at hidden layer
num_classes = 10 # number of output classes discrete range [0,9]
num_epochs = 20 # number of times which the entire dataset is passed throughout the model
batch_size = 1000 # the size of input data took for one iteration
lr = 1e-3 # size of step

train_data = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data  = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

assert len(train_data) == 60000
assert len(test_data)  == 10000


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


train_gen = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_gen  = torch.utils.data.DataLoader(dataset=test_data,  batch_size=batch_size, shuffle=False)

net = Net(input_size, hidden_size, num_classes)

if torch.cuda.is_available():
    net.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

t_start = time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_gen):
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_data)//batch_size}], Loss: {loss.data:.4f}")
print(f"Training took {time()-t_start:.2f}s")

correct = 0
total = 0
for (images, labels) in test_gen:
    images = Variable(images.view(-1, 28*28))
    labels = labels

    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

    output = net(images)
    _, predicted = torch.max(output, 1)
    correct += (predicted == labels).sum()
    total += labels.size(0)

print(f"Accuracy of the model: {(100*correct)/(total+1):.3f}%")

gguf_writer = gguf.GGUFWriter("models/MNIST/mnist-fc-f32.gguf", "mnist-fc")

print("Model tensors saved to models/MNIST/mnist-fc-f32.gguf:")
for tensor_name in net.state_dict().keys():
    data = net.state_dict()[tensor_name].squeeze().numpy()
    print(tensor_name, "\t", data.shape)
    gguf_writer.add_tensor(tensor_name, data)

gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
gguf_writer.close()
