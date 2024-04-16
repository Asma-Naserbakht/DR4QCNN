from qiskit.circuit.library import ZFeatureMap
from utils import QuantumCircuit, conv_layer, pool_layer, train_AE, AE_transform
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.decomposition import PCA
from umap import UMAP
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from data import get_AE_train_data


def get_QCNN(args):
    feature_map = ZFeatureMap(8)

    ansatz = QuantumCircuit(8, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)

    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combining the feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # we decompose the circuit for the QNN to avoid additional data copying
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

    return qnn


def PCA_DR(args, dataloader , pca_model = None):
    # Extract images and targets from the DataLoader
    images, targets = next(iter(dataloader))
    images_flattened = images.view(images.size(0), -1).numpy()

    # Convert all 0s to -1
    targets = targets.clone()
    targets[targets == 0] = -1
    targets = targets.numpy()

    # Apply PCA
    if pca_model is None:
        pca_8d = PCA(n_components=8)
        proj_8d = pca_8d.fit_transform(images_flattened)
    else:
        pca_8d = pca_model
        proj_8d = pca_8d.transform(images_flattened)   
    
    proj_8d = proj_8d.reshape((-1, 1, 8))
    return pca_8d, proj_8d, targets

    
def UMAP_DR (args, dataloader , umap_model = None):  
    # Extract images and targets from the DataLoader
    images, targets = next(iter(dataloader))
    images_flattened = images.view(images.size(0), -1).numpy()

    # Convert all 0s to -1
    targets = targets.clone()
    targets[targets == 0] = -1
    targets = targets.numpy()
    # Apply UMAP
    if umap_model is None:
        umap_8d = UMAP(n_components=8, init='random')
        proj_8d = umap_8d.fit_transform(images_flattened)
    else:
        umap_8d = umap_model
        proj_8d = umap_8d.transform(images_flattened)   
    
    proj_8d = proj_8d.reshape((-1, 1, 8))

    return umap_8d, proj_8d, targets




class Autoencoderv3(nn.Module):
    def __init__(self):
        super(Autoencoderv3, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Dropout2d(p=0.1),
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2, bias=False),
            nn.Dropout2d(p=0.1),
            nn.ReLU(True),
            nn.Conv2d(8, 8, kernel_size=14, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=14, stride=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(8, 4, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(4, 1, kernel_size=5, stride=1, padding=2, bias=False),
        )
        self.rep = None

    def forward(self, x):
        out_en = self.encoder(x)
        self.rep = out_en
        out = self.decoder(out_en)
        return out
    


def AE_DR (args, dataloader , AE_model = None): 
    # Extract images and targets from the DataLoader
    images, targets = next(iter(dataloader))

    # Convert all 0s to -1
    targets = targets.clone()
    targets[targets == 0] = -1
    targets = targets.numpy()

    # Apply AE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)

    if AE_model is None:
        model = Autoencoderv3().to(device)
        subset_loader = get_AE_train_data()
        train_AE(args, model, subset_loader, device)

    else:
        model = AE_model

    proj_8d = AE_transform(model ,images)

    proj_8d = proj_8d.reshape((-1, 1, 8))
    return model, proj_8d.detach(), targets



