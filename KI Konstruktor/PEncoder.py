#Torch ist für neuronale Netzwerke zuständig und zwar 
#torch Mainbibliothek für maschinelles lernen 
#torch.nn Enthält wichtige Funktionen und Klassen, hierdrin ist das Basismodul nn.Module
#torch.nn.functional Funktionsaufrufe und Aktivierungsfunktionen, hat iwas nmit Tensoren zu tun 
#Tensor ist ein Allgemeiner Begriff für Skalar, Vektor und Matrix, also von 0D hoch bis 4D 
#Transformer arbeiten parallel mit Daten anders als RNNs (gehe ich nicht weiter drauf ein)
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# PEncoder oder PositionalEncoding(nn.Modul) ist für die Positionskodierung zuständig, da Transformermodelle keine Reihenfolge oder Sequenzen besitzen
class PEncoder(nn.Module):
    #d_Model steht für die Dimension, max-len für die Länge der Sequenz
    def __init__(self, d_model, max_len = 5000):
        super(PEncoder, self).__init__()
        # ReadMe PE 
        pe = torch.zeros(max_len, d_model)
        # ReadMe Position 
        position = torch.arange(0,max_len, dtype=torch.float).unsqueeze(1)
        # ReadMe div_term
        div_term = torch.exp(torch.arange(0,d_model).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
         # ReadMe RegisterBuffer
        self.register_buffer('pe', pe)

        def forward(self,x):
            return x + self.pe[:x.size(0),:]