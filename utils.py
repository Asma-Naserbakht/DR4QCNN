#all the required functions
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

global objective_func_vals
objective_func_vals = []

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    # plt.title("Objective function value against iteration")
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective function value")
    # plt.plot(range(len(objective_func_vals)), objective_func_vals)
    # plt.show()



def eval(classifier,proj_8d_test,test_targets):
    y_predict_test = classifier.predict(proj_8d_test)
    y_predict_test = y_predict_test.flatten()
    y_test = test_targets.flatten()
    acc = np.mean(y_test==y_predict_test)
    return acc


def train_AE(args, autoencoder, subset_loader, device ):
    # Step 4: Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Step 5: Train the autoencoder
    num_epochs = args.AE_epoch # You can adjust the number of epochs
    for epoch in range(num_epochs):
        for data in subset_loader:
            images, _ = data
            images = images.to(device)

            # Forward pass
            outputs = autoencoder(images)
            loss = criterion(outputs, images)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training completed")



def AE_transform(autoencoder, images):
    autoencoder.eval()  # Set the model to evaluation mode
    _ = autoencoder(images)  # Forward pass to update self.rep
    encoded = autoencoder.rep  # Access the encoded representation

    # Reshape the encoded images
    encoded_reshaped = encoded.view(-1, 1, 8)  # Assuming the encoded output is compatible with this reshaping
    encoded_reshaped= (encoded_reshaped.cpu())

    return encoded_reshaped