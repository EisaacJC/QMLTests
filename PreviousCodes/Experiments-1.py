import pandas as pd
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.datasets import load_iris
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.providers.aer import Aer
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter # Import calibration functions
from sklearn.model_selection import train_test_split
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms import QSVC
from qiskit.providers.aer import noise
from sklearn.metrics import classification_report
import numpy as np
from qiskit.providers.aer import noise
from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS
from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import sklearn.datasets
from sklearn.decomposition import PCA
noise_model = noise.NoiseModel()
def noise(ansatz):
    ansatz_noise=ansatz
    for qubit in range(ansatz.num_qubits):
        ansatz_noise.rx(np.random.rand(), qubit)
        ansatz_noise.ry(np.random.rand(), qubit)
        ansatz_noise.rz(np.random.rand(), qubit)
    return ansatz  



def experiment(xdata,ydata, method, noisemethod, n):
    """
    xdata is a ndarray for features
    ydata is a ndarray for classification value (Only binary classification)
    method refers to the type of experiment, by default is set as VQC 
    noisemethod refers to the action of noise
        feature
        or 
        ansatz
        by default it has a random noise by acting rx and ry with parameter between [0,1]
    n is the number of repetitions
    """
    if method=="VQC":
        precs=[]
        x = xdata[ydata != 2]
        y = ydata[ydata != 2]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        for i in range(0,n):
            feature_map = ZZFeatureMap(feature_dimension=4, reps=1, entanglement="linear")
            ansatz = TwoLocal(num_qubits=4, rotation_blocks=["ry", "rz"], entanglement_blocks="cz")
            #feature_map_noise=noise(feature_map)
            sampler = Sampler()
            sampler.backend = Aer.get_backend("qasm_simulator", noise_model=noise_model, shots=100,
                                                measurement_error_mitigation_cls=CompleteMeasFitter,
                                                measurement_error_mitigation_shots=100)
            vqc1 = VQC(feature_map=feature_map,
            ansatz=ansatz,
            optimizer=SPSA(maxiter=100),
            sampler=sampler)
            if noisemethod=="ansatz":
                ansatz2=noise(ansatz)
                feature_map2=feature_map
            elif noisemethod=="feature":
                ansatz2=ansatz
                feature_map2=noise(feature_map)
            else:
                print("El método no existe")
            vqc2 = VQC(feature_map=feature_map2,
                    ansatz=ansatz2,
                    optimizer=SPSA(maxiter=100),
                    sampler=sampler)
            vqc1.fit(x_train, y_train)
            vqc2.fit(x_train, y_train)
            precs.append(classification_report(y_test, vqc1.predict(x_test)))
            precs.append(classification_report(y_test, vqc2.predict(x_test)))
            print("Iteración:"+ str(i))
        with open("res_bc.txt", 'w') as outfile:
            outfile.writelines((str(i)+'\n' for i in precs))
    elif method=="Torch":
        scores=[]
        X = xdata
        y01=ydata
        y = 2 * y01 - 1
        y01 = 1 * (np.sum(X, axis=1) >= 0)
        X_ = Tensor(X)
        y01_ = Tensor(y01).reshape(len(y)).long()
        y_ = Tensor(y).reshape(len(y), 1)
        num_inputs = 4
        num_samples = 569
        def closure():
            optimizer.zero_grad()  # Initialize/clear gradients
            loss = f_loss(model1(X_), y_)  # Evaluate loss function
            loss.backward()  # Backward pass
            print(loss.item())  # Print loss
            return loss
        for i in range(n):
            print("Iteración",i)
            feature_map = ZZFeatureMap(feature_dimension=num_inputs, reps=1, entanglement="linear")
            ansatz = RealAmplitudes(num_inputs)
            if noisemethod=="ansatz":
                ansatz2=noise(ansatz)
                feature_map2=feature_map
            elif noisemethod=="feature":
                ansatz2=ansatz
                feature_map2=noise(feature_map)
            qc = QuantumCircuit(num_inputs)
            qc.compose(feature_map, inplace=True)
            qc.compose(ansatz, inplace=True)
            # Setup QNN
            qnn1 = EstimatorQNN(
                circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters
            )
            initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn1.num_weights) - 1)
            qnn2 = EstimatorQNN(
                circuit=qc, input_params=feature_map2.parameters, weight_params=ansatz2.parameters
            )

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
            #Model2
            model2 = TorchConnector(qnn2, initial_weights=initial_weights)
            optimizer = LBFGS(model1.parameters())
            f_loss = MSELoss(reduction="sum")
            model2.train()
            optimizer.step(closure)
            y_predict2 = []
            for X, y_target in zip(X, y):
                output = model1(Tensor(X))
                y_predict2 += [np.sign(output.detach().numpy())[0]]
            scores.append(sum(y_predict == y) / len(y))
            scores.append(sum(y_predict2 == y) / len(y))
        pd.DataFrame(scores).to_csv("experimental_data_Torch.csv")
    elif method=="SVC":
        X=xdata
        y=ydata
        scores=[]
        if noisemethod=="feature":
            for i in range(n):
                print("Iteration:",i)
                train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2)
                adhoc_feature_map = ZZFeatureMap(feature_dimension=len(X.T), reps=2, entanglement="linear")
                sampler = Sampler()
                fidelity = ComputeUncompute(sampler=sampler)
                adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)
                qsvc = QSVC(quantum_kernel=adhoc_kernel)
                qsvc.fit(train_features, train_labels)
                qsvc_score = qsvc.score(test_features, test_labels)
                scores.append(qsvc_score)
                pd.DataFrame(scores).to_csv("experimental_data_QSVC.csv")
        else:
            print("Noise method not valid for QSVC")
    else:
        try:
            print("Unable to retrieve a valid method, using VQC as default.")
            experiment(xdata, ydata, "VQC", noisemethod, n)
        except:
            print("An exception occurred")


#ExecutionParameters
#iris_data = load_iris()
#x = iris_data.data
#y = iris_data.target
#x = x[y != 2]
#y = y[y != 2]


#Execution
#ds= sklearn.datasets.load_breast_cancer()

ds=load_iris()
x=ds.data
y=ds.target
x = x[y != 2]
y = y[y != 2]
pca = PCA(n_components=4)

#Execute PCA if the dataset contains more than 4 features, this to be able to simulate on a computer.
if len(x.T)>4:
    x_pca = pca.fit_transform(x)
else:
    x_pca=x

experiment(xdata=x_pca,ydata=y, method=str(input("Select Q-Algorithm\n")), noisemethod=str(input("Select Noise Method\n")), n=int(input("Select the number of iterations\n")))

