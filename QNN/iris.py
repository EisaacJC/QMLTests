import pandas as pd
import numpy as np
from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS
from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from sklearn.datasets import load_iris
import sklearn.datasets

def noise(ansatz):
    ansatz_noise=ansatz
    for qubit in range(ansatz.num_qubits):
        ansatz_noise.rx(np.random.rand(), qubit)
        ansatz_noise.ry(np.random.rand(), qubit)
        ansatz_noise.rz(np.random.rand(), qubit)
    return ansatz

def closure():
    optimizer.zero_grad()  # Initialize/clear gradients
    loss = f_loss(model1(X_), y_)  # Evaluate loss function
    loss.backward()  # Backward pass
    print(loss.item())  # Print loss
    return loss
scores=[]
for i in range(27):
    ds = load_iris()
    x = ds.data
    y = ds.target
    X = x[y != 2]
    y = y[y != 2]
    y01=y
    y = 2 * y01 - 1
    y01 = 1 * (np.sum(X, axis=1) >= 0)
    X_ = Tensor(X)
    y01_ = Tensor(y01).reshape(len(y)).long()
    y_ = Tensor(y).reshape(len(y), 1)
    num_inputs = 4
    num_samples = len(X)
    feature_map = ZZFeatureMap(feature_dimension=num_inputs, reps=1, entanglement="linear")
    #feature_map=noise(feature_map)
    ansatz = RealAmplitudes(num_inputs)
    qc = QuantumCircuit(num_inputs)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    # Setup QNN
    qnn1 = EstimatorQNN(circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters)
    initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn1.num_weights) - 1)
                #Model1
    model1 = TorchConnector(qnn1, initial_weights=initial_weights)
    optimizer = LBFGS(model1.parameters())
    f_loss = MSELoss(reduction="sum")
    model1.train()
    optimizer.step(closure)
    y_predict = []
    for X, y_target in zip(X, y):
        output = model1(Tensor(X))
        y_predict += [np.sign(output.detach().numpy())[0]]
    scores.append(sum(y_predict == y01) / len(y01))
pd.DataFrame(scores).to_csv("SinRuidoQNN_iris.csv")