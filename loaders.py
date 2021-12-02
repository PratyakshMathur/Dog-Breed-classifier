from imports import *

##### Pre-Processing
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 5
num_workers = 0

data_dir = 'dogImages/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')


###### Normalisation

standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                              
                                              
data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),standard_normalization]),
'test': transforms.Compose([transforms.Resize(size=(224,224)),transforms.ToTensor(),standard_normalization])
} 
                                      
                                     
                                            
###### Image dataset loading

                  
train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,num_workers=num_workers,shuffle=False)
                                            
                                           
                                                                                    
                                           
loaders_transfer = {
    'train': train_loader,
    'test': test_loader
}

