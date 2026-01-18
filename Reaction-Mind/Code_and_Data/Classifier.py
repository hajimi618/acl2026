import json
import copy
import random
import pandas as pd
import numpy as np
from sklearn import metrics
import random
from sklearn.linear_model import LogisticRegression
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from matplotlib import pyplot as plt


##################################################################################
# Train
##################################################################################

with open('/rxnfp-master/data/rxnclass2id.json', 'r') as f:
    rxnclass2id = json.load(f)

with open('/rxnfp-master/data/rxnclass2name.json', 'r') as f:
    rxnclass2name = json.load(f)
all_classes = sorted(rxnclass2id.keys())


import pickle
schneider_df = pd.read_csv('/data/schneider50k.tsv', sep='\t', index_col=0)
# ft_10k_fps = np.load('/rxnfp-master/data/fps_ft_10k.npz')['fps']
ft_10k_fps = np.load('/data/schneider50k_rxnfp_reactants_full_fingerprint_before_int8.npz')['fingerprints']
# ft_pretrained = np.load('/rxnfp-master/data/fps_pretrained.npz')['fps']
schneider_df['ft_10k'] = [fp for fp in ft_10k_fps]
# schneider_df['ft_pretrained'] = [fp for fp in ft_pretrained]
schneider_df['class_id'] = [rxnclass2id[c] for c in schneider_df.rxn_class]
schneider_df.head()


import pdb; pdb.set_trace()
train_df = schneider_df[schneider_df.split=='train']
test_df = schneider_df[schneider_df.split=='test']
print(len(train_df), len(test_df))

# lr_cls =  LogisticRegression(max_iter=1000)
scrambled_train_rxn_ids = [rxnclass2id[c] for c in train_df.rxn_class]
test_rxn_class_ids = [rxnclass2id[c] for c in test_df.rxn_class]

X_train = np.stack(train_df.ft_10k.values).astype(np.float32)
Y_train = train_df.class_id.values

X_val = np.stack(test_df.ft_10k.values).astype(np.float32)
Y_val = test_df.class_id.values

X_tensor = torch.from_numpy(X_train)
Y_tensor = torch.from_numpy(Y_train).long()

X_tensor_test = torch.from_numpy(X_val)
Y_tensor_test = torch.from_numpy(Y_val).long()

batch_size = 256
train_dataset = TensorDataset(X_tensor, Y_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_tensor_test, Y_tensor_test)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



class MLP_3_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP_3_Layer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8192)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8192, 2048)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x



class MLP_2_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP_2_Layer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP_3_Layer(input_dim=10240, hidden_dim=8192, num_classes=50).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

num_epochs = 500

best_val_acc = 0.0
patience = 20
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_dataloader:

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.2f}%")

    model.eval()
    valid_loss = 0.0
    correct = 0
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            topk_vals, topk_preds = outputs.topk(3, dim=1, largest=True, sorted=True)

            # Top-1
            top1_preds = topk_preds[:, 0]
            top1_correct += top1_preds.eq(labels).sum().item()

            # Top-2
            in_top2 = labels.view(-1, 1).eq(topk_preds[:, :2]).any(dim=1)
            top2_correct += in_top2.sum().item()

            # Top-3
            in_top3 = labels.view(-1, 1).eq(topk_preds).any(dim=1)
            top3_correct += in_top3.sum().item()

    val_loss = valid_loss / len(train_dataset)

    top1_acc = 100. * top1_correct / total
    top2_acc = 100. * top2_correct / total
    top3_acc = 100. * top3_correct / total

    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f} | Top-1 Acc: {top1_acc:.2f}% | Top-2 Acc: {top2_acc:.2f}% | Top-3 Acc: {top3_acc:.2f}%")

    if top3_acc > best_val_acc:
        best_val_acc = top3_acc
        epochs_no_improve = 0
        # torch.save(model.state_dict(), 'Best_Model_MLP.pt')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_val_acc:.2f}%")
            break






##################################################################################
# Test
##################################################################################


import json
import copy
import random
import pandas as pd
import numpy as np
from sklearn import metrics
import random
from sklearn.linear_model import LogisticRegression
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit import DataStructs
from rdkit.Chem.rdmolops import LayeredFingerprint, PatternFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import rdMolDescriptors, RDKFingerprint



def split_smiles(smiles):
    
    label = smiles.strip()
    parts = label.split(">")
    
    if (len(parts[1]) == 0):
        reactants_reagents = parts[0].split(".")
    else:
        reactants_reagents = parts[0].split(".") + parts[1].split(".")
    
    reactants_reagents_str = ".".join(reactants_reagents)

    return reactants_reagents_str



def compute_reaction_side_fingerprint(smiles_list, fp_size=2048, radius=2):

    fps_concat = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        import pdb; pdb.set_trace()
        if mol is None:
            fps_concat.append(np.zeros(fp_size * 7, dtype=int))
            continue

        # 1. RDKFingerprint
        rdkit_fp = RDKFingerprint(mol, fpSize=fp_size)
        arr_rdk = np.zeros((fp_size,), dtype=int)
        ConvertToNumpyArray(rdkit_fp, arr_rdk)

        # 2. LayeredFingerprint
        layered_fp = LayeredFingerprint(mol, fpSize=fp_size)
        arr_layer = np.zeros((fp_size,), dtype=int)
        ConvertToNumpyArray(layered_fp, arr_layer)

        # 3. PatternFingerprint
        pattern_fp = PatternFingerprint(mol, fpSize=fp_size)
        arr_pattern = np.zeros((fp_size,), dtype=int)
        ConvertToNumpyArray(pattern_fp, arr_pattern)

        # 4. Avalon
        avalon_fp = GetAvalonFP(mol, nBits=fp_size)
        arr_avalon = np.zeros((fp_size,), dtype=int)
        ConvertToNumpyArray(avalon_fp, arr_avalon)

        # 5. AtomPair
        """ap_fp = GetHashedAtomPairFingerprintAsBitVect(mol, fpSize=fp_size)
        arr_ap = np.zeros((fp_size,), dtype=int)
        ConvertToNumpyArray(ap_fp, arr_ap)"""

        # 6. Topological Torsion
        """tt_fp = GetHashedTopologicalTorsionFingerprintAsBitVect(mol, fpSize=fp_size)
        arr_tt = np.zeros((fp_size,), dtype=int)
        ConvertToNumpyArray(tt_fp, arr_tt)"""

        # 7. Morgan
        morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, fp_size)
        arr_morgan = np.zeros((fp_size,), dtype=int)
        ConvertToNumpyArray(morgan_fp, arr_morgan)

        concatenated = np.concatenate([arr_rdk,
                                       arr_layer,
                                       arr_pattern,
                                       arr_avalon,
                                       arr_morgan,
                                       ])
        fps_concat.append(concatenated)
    return np.vstack(fps_concat)



class MLP_3_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP_3_Layer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8192)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8192, 2048)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        emb = self.fc2(x)
        relu_emb = self.relu2(emb)
        out = self.fc3(relu_emb)
        return out, emb



class MLP_2_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP_2_Layer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        emb = self.fc1(x)
        relu_emb = self.relu1(emb)
        out = self.fc2(relu_emb)
        return out, emb
    


def predict_topk(model, input_batch_np, topk=3):

    model.eval()
    device = next(model.parameters()).device

    input_tensor = torch.from_numpy(input_batch_np).float().to(device)

    with torch.no_grad():
        outputs, embeddings = model(input_tensor)

        x0 = embeddings[0].unsqueeze(0)
        other_x = embeddings[1:]
        l2_dist = torch.norm(other_x - x0, dim=1)
        print("Similarities between x[0] and others:", l2_dist)
        import pdb; pdb.set_trace()

        probs = torch.softmax(outputs, dim=1)
        topk_probs, topk_preds = probs.topk(topk, dim=1)

    topk_preds = topk_preds.cpu().numpy()
    topk_probs = topk_probs.cpu().numpy()

    results = []
    for i in range(len(input_batch_np)):
        topk_result = [(int(cls), float(conf)) for cls, conf in zip(topk_preds[i], topk_probs[i])]
        results.append(topk_result)

    return results



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_3_Layer(10240, 8192, 50).to(device)

    model.load_state_dict(torch.load("Best_Model_MLP.pt", map_location=device))

    smiles_list = ["CC(C=CC(=O)O)Oc1ccc(Oc2ccc(C(F)(F)F)cc2)cc1.CC(C)(C)C(=O)Cl.O=C([O-])CC(=O)[O-].CCC(C)(CC)O[Mg+2].CCOC(=O)CC(=O)OCC",
                   "[Na].CC(C)=O.Cl.CCCCCCC(=N)N.Fc1ccc(N=C=S)cc1.c1ccccc1.CCCCC"]

    fun_encoded_batch = compute_reaction_side_fingerprint(smiles_list, fp_size=2048, radius=2)

    top3_results = predict_topk(model, fun_encoded_batch, topk=3)

    for i, topk in enumerate(top3_results):
        print(f"Sample {i+1}:")
        for rank, (cls, conf) in enumerate(topk, 1):
            print(f"  Top-{rank} Class = {cls}, Confidence = {conf:.4f}")
