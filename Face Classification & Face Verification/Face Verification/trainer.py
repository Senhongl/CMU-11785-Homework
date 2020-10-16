from torch import optim
from sklearn.metrics import roc_auc_score
import sys
import torch
from torch import nn
import pandas as pd
from data import train_datasets, val_datasets
from triplet_loss import TripletLoss
from model import Network

def get_auc(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    return auc

def train_verification(model, test_loader, images_per_person):
    model.train()

    for epoch in range(numEpochs):
        train_data = train_datasets(root='./train_data/medium', transform = data_transform, images_per_person = images_per_person) + train_datasets(root='./train_data/large', transform = data_transform, images_per_person = images_per_person, start = 2300)

        train_loader_args = dict(batch_size= 10, pin_memory=True, shuffle = True, num_workers = 8) 
        train_loader = Data.DataLoader(train_data, **train_loader_args)
        start = time.time()
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(train_loader):
            labels = np.repeat(labels, images_per_person)
            d0, d1, d2, d3, d4 = feats.shape
            feats = feats.reshape(d0 * d1, d2, d3, d4)
            
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(feats)
            outputs = F.avg_pool2d(outputs, [outputs.size(2), outputs.size(3)], stride=1)
            outputs = F.normalize(outputs, p = 2, dim = 1)
            outputs = outputs.reshape(outputs.shape[0], outputs.shape[1])
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

            if batch_num % 10 == 9:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tRunningtime: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/10, time.time() - start))
                avg_loss = 0.0  
            
            del outputs
            del feats
            del labels
            del loss
            torch.cuda.empty_cache()
        

        torch.save(model.state_dict(), './model_verification_{}.pt'.format(epoch))




def test_verify(model, test_loader):
    model.eval()
    
    score = []
    labels = []
    for batch_num, (feats, label) in enumerate(test_loader):
        feats = feats.to(device)
        feats_A = feats[:, :3] # Face embedding of first images
        feats_B = feats[:, 3:] # Face embedding of second images
        
        outputs_A = model(feats_A)
        outputs_B = model(feats_B)
        outputs_A = F.avg_pool2d(outputs_A, [outputs_A.size(2), outputs_A.size(3)], stride=1)
        outputs_A = outputs_A.reshape(outputs_A.shape[0], outputs_A.shape[1])
        outputs_B = F.avg_pool2d(outputs_B, [outputs_B.size(2), outputs_B.size(3)], stride=1)
        outputs_B = outputs_B.reshape(outputs_B.shape[0], outputs_B.shape[1])
        
        outputs_A = outputs_A.cpu() 
        outputs_B = outputs_B.cpu()
        
        tmp = torch.sum((outputs_A * outputs_B), dim = 1) / (torch.pow(torch.sum(torch.pow(outputs_A, 2), dim = 1), 0.5) * torch.pow(torch.sum(torch.pow(outputs_B, 2), dim = 1), 0.5))


        score.extend(tmp.cpu().detach().numpy())
        labels.extend(label.numpy())
        
        del tmp
        del feats_A
        del feats_B
        del outputs_A
        del outputs_B
        torch.cuda.empty_cache()
        
    return score, labels

if __name__ == 'main':
	numEpochs = 1000
	num_feats = 3
	closs_weight = 1
	feat_dim = 2300

	learningRate = 1e-2


	hidden_sizes = [32, 64, 128, 256]
	num_classes = 2300

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	network = Network(num_feats, hidden_sizes, num_classes)
	cuda = torch.cuda.is_available()
	device = torch.device("cuda" if cuda else "cpu")

	network.linear_label = nn.Linear(256, 4600)
	network.linear_closs = nn.Linear(256, 4600)

	val_data = val_datasets('validation_trials_verification.txt', 'validation_verification', data_transform_val)

	val_loader_args = dict(shuffle=False, batch_size= 256, pin_memory=True, num_workers = 8) 
	val_loader = Data.DataLoader(val_data, **val_loader_args)
	numEpochs = 100
	model = nn.Sequential(*list(network.children())[:-3])
	criterion = TripletLoss(margin = 0.7)
	optimizer = optim.Adam(model.parameters(), lr = 0.0001)
	model.train()
	model.to(device)
	train_verification(model, val_loader, 40)
