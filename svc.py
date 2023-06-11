from qiskit.utils import algorithm_globals
from sklearn.datasets import load_iris
algorithm_globals.random_seed = 12345
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import sklearn.datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


iris_data = load_iris()
X = iris_data.data
y = iris_data.target
#x=bc.data
#pca = PCA(n_components=4)
#x_pca = pca.fit_transform(x)
#y=bc.target
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2)
adhoc_feature_map = ZZFeatureMap(feature_dimension=len(X.T), reps=2, entanglement="linear")
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)
#adhoc_score_callable_function = adhoc_svc.score(test_features, test_labels)
from qiskit_machine_learning.algorithms import QSVC
qsvc = QSVC(quantum_kernel=adhoc_kernel)
qsvc.fit(train_features, train_labels)
qsvc_score = qsvc.score(test_features, test_labels)
print(f"QSVC classification test score: {qsvc_score}")
