import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

select=int(input('Enter 1 - FC or 2 - CNN : ')) # For choosing between FC and CNN models

class Autoencoder(nn.Module):
  #FC Autoencoder 
  if(select==1):
    def __init__(self):
        super().__init__()
        # Encoder layer
        self.en1 = nn.Linear(784, 256)
        self.en2 = nn.Linear(256, 128)
        # Decoder layers
        self.de2 = nn.Linear(128, 256)
        self.de1 = nn.Linear(256, 784)
    def forward(self, X):
        #Encoding
        X = F.relu(self.en1(X))
        X = F.relu(self.en2(X))
        #Decoding
        X = F.relu(self.de2(X))
        X = F.relu(self.de1(X))
        return X

  elif(select==2):
    # CNN Autoencoder
      def __init__(self):
          super().__init__()
        
          #Encoder layers
          self.conv1 = nn.Conv2d(1, 4, 3, padding=1)  
          self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
          self.pool = nn.MaxPool2d(2, 2)
        
          #Decoder layers
          self.tconv1 = nn.ConvTranspose2d(8, 4, 3 , padding=1)  
          self.tconv2 = nn.ConvTranspose2d(4, 1, 3,  padding=1)  
          self.tconv3= nn.ConvTranspose2d(1, 1, 3, padding=1) 
          
          #Upsmapling layer
          self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

      def forward(self, X):
          #Encoding
          X = F.relu(self.conv1(X))
          X = self.pool(X)
          X = F.relu(self.conv2(X))
          X = self.pool(X)
          #Decoding
          X = F.relu(self.tconv1(X))
          X = F.relu(self.upsample(X))
          X = F.relu(self.tconv2(X))
          X = F.relu(self.upsample(X))
          X = torch.sigmoid(self.tconv3(X))
          return X
  else:
    print('Invalid selection, try again')
    quit()

# Load MNIST datasets for training and testing
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./dataset', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./dataset', train=False, transform=transform)
train_loader = DataLoader(train_dataset,  100, shuffle=True) 
test_loader = DataLoader(test_dataset,  100, shuffle=False) 

# Intialize the model
model=Autoencoder()
print(model)

# Using MSEloss  
criterion = nn.MSELoss()

#Using Adam opt with lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

epochs = 10
# Train Model 
for epoch in range(1, epochs+1):
    train_loss = 0.0
    #Training
    for data in train_loader:
        images, _ = data
        if(select==1):
          images = images.view(-1, 28*28).requires_grad_()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)      
    train_loss /= len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.5f}'.format(epoch, train_loss))


test_sample1 = []
test_sample2 = []
# Generating test samples 
iter= iter(test_loader)
test_images, labels = iter.next()
i=0
for j in range(10):
  while(labels[i]!=j):
    i+=1
  test_sample1.append(i)
k=i-1
j=0
for j in range(10):
  while(labels[k]!=j):
    k-=1
  test_sample2.append(k)

# Applying model on test images
if(select==1):
  output =  model(test_images.view(-1,28*28))
else:
  output =  model(test_images)

# Generating images and ploting 
def gen_output(test_sample,Fimg):
  temp = 0
  for sample in test_sample:
    temp+=1
    img = test_images[sample].numpy().reshape(28, 28)
    plt.subplot(2,10,temp)
    plt.imshow(img,cmap='gray')
    plt.axis('off') 
    img = torch.reshape(output[sample],(28,28))
    img = img.detach().numpy()
    plt.subplot(2,10,temp+10)
    plt.imshow(img,cmap='gray')
    plt.axis('off')
    plt.savefig(Fimg)

if(select==1):
  gen_output(test_sample1,'FC_1.jpg')
  gen_output(test_sample2,'FC_2.jpg')
else :
  gen_output(test_sample1,'CNN_1.jpg')
  gen_output(test_sample2,'CNN_2.jpg')



