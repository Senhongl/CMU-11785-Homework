from util import data_transform, data_transform_valfrom util 
import torchvision

data_transform = transforms.Compose([                           
    transforms.RandomHorizontalFlip(),                        
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[    
                             0.229, 0.224, 0.225])
         ])

data_transform_val = transforms.Compose([                                                   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[    
                             0.229, 0.224, 0.225])
         ])

