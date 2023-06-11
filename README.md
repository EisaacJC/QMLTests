# QMLTests
QML Tests, in particular Main_Test is a script designed to test how well sampler from Qiskit behaves when feature map or ansatz proposal are disturbed with a random configuration of quantum gates with a random probability between [0,1], this corresponds to an arbitrary change that oscillates between 0-15% in each direction, since the parameters are well determined from [0 , $$\pi$$ ].
# About the code.
This code has the intention to evaluate if a modification on the ansatz or the feature map has a significant
impact on the value of prediction for a determined dataset.
The noise method that is set in this experiment is related with a random parametrization on three different
directions, this is set on an interval from [0,1], this represents an error that oscillates between [0%, 15.91%] in each direction.

The code evaluates the behavior of the algorithm in the presence and in the absence of noise.
# About the options.
This code has a collection of essential functions that create an arbitrary modification of the behaviour of the code.

## Q-Algorithms

In this code we present a collection of codes that are modified from the QML Tutorials by Qiskit, here we present
a collection of three different algorithsms:

    a) VQC Variational Quantum Classifier
    b) Torch method for optimize QNN
    c) Quantum Support Vectorial Classification

Each of this algorithms can be called by th program. In particular for VQC is possible to extract multiple metrics related 
to precision, f1 score and recall.

## Noise Method

One of the main questions of this project was related to discover if the impact of random noise in the feature map or in the ansatz
have significant impact on the metrics, so we manage to modify this by modifying one of this at a time to find if the measure of one 
can improve or affect the function of the Q-Algorithm.

    a) feature, (string) this method is related to edit the feature map mantaining the ansatz without noise.
    b) ansatz, (string) this method of noise is related to edit the ansatz and mantaining fm without modifications.

# Number of iterations

To obtain reliable data we set a numeric expression that represents the number of executions of this algorithm to  obtain
statistical significance if there is. 
    
    n parameter (int)

# Copyright Disclaimer
This code is part of Qiskit.
(C) Copyright IBM 2017.
This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
This is a modification of QML algorithms by Qiskit and IBM, this is intended only for educational purpose
Any doubts about content please refer to eisaacjc8@gmail.com
