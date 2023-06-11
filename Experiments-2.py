import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.providers.aer import Aer
from qiskit.visualization import plot_histogram # Import plot_histogram function
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter # Import calibration functions
from sklearn.model_selection import train_test_split
from qiskit.primitives import Sampler
import matplotlib.pyplot as plt 
from qiskit.providers.aer import noise
from sklearn.metrics import classification_report
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
    if method=="default":
        precs=[]
        x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.05)
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
                    #warm_start=True,
                    sampler=sampler)

            vqc1.fit(x_train, y_train)
            vqc2.fit(x_train, y_train)

            cal_circuits, state_labels = complete_meas_cal(qr=vqc1.circuit.num_qubits,
                                                        circlabel='measurement_calibration')
            cal_circuits2, state_labels2 = complete_meas_cal(qr=vqc2.circuit.num_qubits,
                                                        circlabel='measurement_calibration')
            precs.append(classification_report(y_test, vqc1.predict(x_test)))
            precs.append(classification_report(y_test, vqc2.predict(x_test)))
            #precs.append("hola")
            print("Iteración:"+ str(i))
    with open("res_bc_feature.txt", 'w') as outfile:
        outfile.writelines((str(i)+'\n' for i in precs))
#iris_data = load_iris()
#x = iris_data.data
#y = iris_data.target
#x = x[y != 2]
#y = y[y != 2]

import sklearn as sk
import sklearn.datasets

bc= sklearn.datasets.load_breast_cancer()
x=bc.data
y=bc.target
# Importar PCA
from sklearn.decomposition import PCA

# Crear un objeto PCA con 2 componentes
pca = PCA(n_components=4)

# Ajustar y transformar el dataset
x_pca = pca.fit_transform(x)


experiment(xdata=x_pca,ydata=y, method="default", noisemethod="feature", n=27)

