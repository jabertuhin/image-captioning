import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.word_embedings = nn.Embedding(vocab_size, embed_size)
        
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.2, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        
        self.init_weights()
                
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.word_embedings.weight)
        
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        captions = self.word_embedings(captions)
        
        features = features.unsqueeze(1)
        
        inputs = torch.cat((features, captions), 1)
        out, _ = self.lstm(inputs)
        
        out = self.fc(out)
        
        return out

    def sample(self, inputs, states=None, max_len=20):                
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_words = []
        
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        for i in range(max_len):            
            output, hidden = self.lstm(inputs, hidden)            
                                    
            output = self.fc(output)
            output = output.squeeze(1)
            word_id  = output.argmax(dim=1)            
            
            predicted_words.append(word_id.item())
            
            inputs = self.word_embedings(word_id.unsqueeze(0))
            
        return predicted_words
    
    
    
    
    
    
    
    
    