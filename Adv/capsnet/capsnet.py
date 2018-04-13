
# coding: utf-8

# In[1]:


# imports 
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable
from torchvision import datasets,transforms
import torch.nn.functional as F
import torchvision.utils as vutils


# ## Capsule Conv Layer

# In[2]:


class CapsuleConvLayer(nn.Module) :
    def __init__(self,in_channels,out_channels):
        super(CapsuleConvLayer,self).__init__()
        
        self.conv0 = nn.Conv2d(in_channels=in_channels,
                              out_channels = out_channels,
                              kernel_size = 9,
                              stride=1,
                              bias=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        return self.relu(self.conv0(x))


# ## Capsule Layer

# In[3]:


class ConvUnit(nn.Module):
    def __init__(self,in_channels):
        super(ConvUnit,self).__init__()
        
        self.conv0 = nn.Conv2d(in_channels=in_channels,
                              out_channels=32,
                              kernel_size=9,
                              stride=2,
                              bias=True)
    def forward(self,x):
        return self.conv0(x)


# In[4]:


class CapsuleLayer(nn.Module):
    def __init__(self,in_units,in_channels,
                num_units,unit_size,
                use_routing):
        super(CapsuleLayer,self).__init__()
        
        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing
        
        if self.use_routing :
            self.W = nn.Parameter(torch.randn(1,
                                             in_channels,
                                             num_units,
                                             unit_size,
                                             in_units))
        else :
            def create_conv_unit(unit_id):
                unit = ConvUnit(in_channels=in_channels)
                self.add_module("unit_"+str(unit_id),
                               unit)
                return unit
            
            self.units = [create_conv_unit(i) for i in range(self.num_units)]
    
    @staticmethod
    def squash(s):
        #Eq 1 from paper
        mag_s_j = torch.sum(s**2,dim=2,keepdim=True)
        mag = torch.sqrt(mag_s_j)
        
        s = (mag_s_j/(1+mag_s_j))*(s/mag)
        return s
    
    def forward(self,x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)
    
    def no_routing(self,x):
        #[batch,channels,height,width]

        #get output for each unit.
        u = [self.units[i](x) for i in range(self.num_units)]
        
        #stack unit outputs
        #[batch,unit,channels,height,width]
        u = torch.stack(u,dim=1)
        
        #Flatten to [batch,unit,output]
        u = u.view(x.size(0),self.num_units,-1)
        
        #return squashed outputs.
        return CapsuleLayer.squash(u)
    
    
    def routing(self,x):
        batch_size = x.size(0)
        
        #[batch,in_units,features] -> [batch,features,in_units]
        x =x.transpose(1,2)
        
        #Wdotx  = [batch,features,num_units,    unit_size,in_units] dot [batch,features,num_units,    in_units,1]
        #output Wdotx = [batch,features,num_units,   unit_size,1]
        
        #[batch,features,in_units] -> [batch,features,num_units,in_units,1]
        x = torch.stack([x]*self.num_units,dim=2).unsqueeze(4)
        
        # [batch,features,num_units,unit_size,in_units]
        W = torch.cat([self.W]*batch_size,dim=0)
            
        #[batch,features,num_units,   unit_size,1]
#         print W.shape
#         print x.shape
        u_hat= torch.matmul(W,x)
        
        #routing logits.
        #[batch,features,num_units,1]
        if torch.cuda.is_available():
            b_ij = Variable(torch.zeros(1,self.in_channels,
                                   self.num_units,1)).cuda()
        else :
            b_ij = Variable(torch.zeros(1,self.in_channels,
                                   self.num_units,1))
            
        
        num_iterations = 3
        
        for iteration in range(num_iterations):
            #routing_logits->softmax
            #[batch,features,num_units,1,1]
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij]*batch_size,dim=0).unsqueeze(4)
            
            #apply routing to weighted_inputs u_hat
            #[batch,1,num_units,unit_size,1]
            s_j = (c_ij*u_hat).sum(dim=1,keepdim=True)
            
            #squash
            #[batch,1,num_units,unit_size,1]
            v_j = CapsuleLayer.squash(s_j)
            
            #[batch,features,num_units,unit_size,1]
            v_j1 = torch.cat([v_j]*self.in_channels,dim=1)
            
            #batch,features,num_units,1,1 after matmul
            #batch,features,num_units,1 after squeeze
            #1,features,num_units,1 after mean
            u_vj1 = torch.matmul(u_hat.transpose(3,4),v_j1).squeeze(4).mean(dim=0,keepdim=True)
            
            #update routing 
            b_ij = b_ij + u_vj1
            
            
        return v_j.squeeze(1)
            
            
        


# # FINAL CapsuleNetwork

# In[5]:


class CapsuleNetwork(nn.Module):
    def __init__(self,image_width,image_height,
                image_channels,
                conv_inputs,conv_outputs,
                num_primary_units,
                primary_unit_size,
                num_output_units,
                output_unit_size):
        super(CapsuleNetwork,self).__init__()
        
        self.reconstructed_image_count = 0
        
        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height
        
        
        self.conv1 = CapsuleConvLayer(in_channels=conv_inputs,
                                     out_channels=conv_outputs)
        
        self.primary = CapsuleLayer(in_units=0,
                                   in_channels=conv_outputs,
                                   num_units=num_primary_units,
                                   unit_size=primary_unit_size,
                                   use_routing=False)
        
        self.digits = CapsuleLayer(in_units=num_primary_units,
                                  in_channels=primary_unit_size,
                                  num_units=num_output_units,
                                  unit_size=output_unit_size,
                                  use_routing=True)
        
        reconstruction_size = image_width*image_height*image_channels
        
        self.reconstruct0 = nn.Linear(num_output_units*output_unit_size,
                                     int((reconstruction_size*2)/3))
        self.reconstruct1 = nn.Linear(int((reconstruction_size*2)/3),
                                     int((reconstruction_size*3)/2))
        self.reconstruct2 = nn.Linear(int(reconstruction_size*3)/2,
                                     reconstruction_size)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self,x):
        return self.digits(self.primary(self.conv1(x)))
    
    def loss(self,images,input,target,size_average=True):
        return self.margin_loss(input,target,size_average) + self.reconstruction_loss(images,input,size_average)
    
    
    def margin_loss(self,input,target,size_average):
        
        batch_size=input.size(0)
        
        # ||vc|| from paper.
        v_mag = torch.sqrt((input**2).sum(dim=2,keepdim=True))
        
        #calculate left and right max() terms. eq4 paper.
        if torch.cuda.is_available():
            zero = Variable(torch.zeros(1)).cuda()
        else :
            zero = Variable(torch.zeros(1))
        m_plus = 0.9
        m_minus = 0.1
        
        max_l = torch.max(m_plus - v_mag,zero).view(batch_size,-1)**2
        max_r = torch.max(v_mag - m_minus,zero).view(batch_size,-1)**2
        
        #eq4
        loss_lambda = 0.5
        T_c = target
        L_c = (T_c * max_l) + (loss_lambda * (1-T_c)*max_r)
        
        if size_average :
            L_c = L_c.mean()
            
        return L_c
    
    def reconstruction_loss(self,images,input,size_average=True):
        #lengths of capsules.
        v_mag = torch.sqrt((input**2).sum(dim=2))
        
        #idx of longest capsule output.
        _,v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data
        
        #use this capsules representation to reconstruct image.
        batch_size = input.size(0)
        all_masked = [None]*batch_size
        
        for batch_idx in range(batch_size):
            #sample from batch.
            input_batch = input[batch_idx]
            
            #copy max capsules idx from this sample.
            #leave other caps as zero.
            if torch.cuda.is_available():
                batch_masked = Variable(torch.zeros(input_batch.size())).cuda()
            else :
                batch_masked = Variable(torch.zeros(input_batch.size()))
                
            batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
            all_masked[batch_idx] = batch_masked
        
        #stack masked caps over batch dimen.
        masked = torch.stack(all_masked,dim=0)
        
        #reconstruction.
        masked = masked.view(input.size(0),-1)
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))
        
        output = output.view(-1,self.image_channels,
                            self.image_height,
                            self.image_width)
        
        if self.reconstructed_image_count%10 == 0:
            if output.size(1) == 2:
                #handle 2 channel images.
                
                zeros = torch.zeros(output.size(0),-1,output.size(2),output.size(3))
                output_image = torch.cat([zeros,output.data],dim=1).cpu()
            else:
                output_image = output.data.cpu()
            vutils.save_image(output_image,"reconstructed_images/recontruction_"+str(self.reconstructed_image_count)+".png")
        
        self.reconstructed_image_count +=1
        
        #loss is sum sqr. diff. between input and reconstructed image.
        error = (output-images).view(output.size(0),-1)
        error = error**2
        error = torch.sum(error,dim=1)*0.0005

        if size_average:
            error = error.mean(dim=0)

        
        return error


# ### HyperParams

# In[6]:


learning_rate = 0.01
batch_size = 128
test_batch_size = 128
early_stop_loss = 0.0001


# In[7]:


dataset_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),(0.3081,))])


# In[8]:


train_dataset = datasets.MNIST('./data',
                              train=True,
                              download=True,
                              transform=dataset_transform)


# In[9]:


train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)


# In[10]:


test_dataset = datasets.MNIST('./data',
                             train=False,
                             download=True,
                             transform=dataset_transform)


# In[11]:


test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=test_batch_size,shuffle=True)


# ## Create CapsNet

# In[12]:


conv_inputs=1
conv_outputs=256

num_primary_units=8
primary_unit_size=32*6*6

output_unit_size=16


# In[13]:


network = CapsuleNetwork(image_width=28,
                        image_height=28,
                        image_channels=1,
                        conv_inputs=conv_inputs,
                        conv_outputs=conv_outputs,
                        num_primary_units=num_primary_units,
                        primary_unit_size=primary_unit_size,
                        num_output_units=10,
                        output_unit_size=output_unit_size)

if torch.cuda.is_available():
    network = network.cuda()
print network


# In[14]:


def to_one_hot(x,length):
    batch_size = x.size(0)
    
    x_one_hot = torch.zeros(batch_size,length)
    for i in range(batch_size):
        x_one_hot[i,x[i]] = 1.0
    return x_one_hot


# ## MNIST TEST FUNCTION

# In[15]:


def test():
    network.eval()
    
    test_loss =0
    correct = 0
    
    for data,target in test_loader:
        target_indices = target
        target_one_hot = to_one_hot(target_indices,length=network.digits.num_units)
        
        if torch.cuda.is_available():
            data,target = Variable(data,volatile=True).cuda() , Variable(target_one_hot).cuda()
        else:
            data,target = Variable(data,volatile=True) , Variable(target_one_hot)
        
        output=network(data)
        
        test_loss += network.loss(data,output,target,size_average=True).data[0]
        
        v_mag = torch.sqrt((output**2).sum(dim=2,keepdim=True))
        
        pred = v_mag.data.max(1,keepdim=True)[1].cpu()
        
        correct += pred.eq(target_indices.view_as(pred)).sum()
        
    test_loss /= len(test_loader.dataset)
    
    print 'Test Set: Average Loss: {:.4f},Accuracy :{}/{} ({:.0f}%)'.format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset))
    


# In[16]:


def train(epoch):
    optimizer = optim.Adam(network.parameters(),lr=learning_rate)
    
    last_loss = None
    log_interval = 20
    
    network.train()
    
    for batch_idx , (data,target) in enumerate(train_loader):
        target_one_hot = to_one_hot(target,length=network.digits.num_units)
        
        if torch.cuda.is_available():
            data,target = Variable(data).cuda(),Variable(target_one_hot).cuda()
        else:
            data,target = Variable(data),Variable(target_one_hot)
            
        optimizer.zero_grad()
        
        output=network(data)
        
        loss = network.loss(data,output,target)
#         print loss.shape
        loss.backward()
        
        last_loss = loss.data[0]
        
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print 'Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(epoch,
                                                                            batch_idx*len(data),
                                                                            len(train_loader.dataset),
                                                                            100.*batch_idx/len(train_loader),
                                                                            loss.data[0])
        if last_loss < early_stop_loss :
            print 'Breaking due to early_stop_loss inside func-train'
            break
    return last_loss


# # RUN

# In[17]:

import time

num_epochs = 10
for epoch in range(1,num_epochs+1):
    start = time.time()
    last_loss = train(epoch)
    end = time.time()
    print 'Time Taken for epoch:',epoch,':',(end-start)/60,'minutes'
    torch.save(network.state_dict(),'network_'+str(epoch)+'.pkl')
    test()

    
    if last_loss < early_stop_loss:
        print 'Breaking due to early_stop_loss'
        break

