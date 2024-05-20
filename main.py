import pandas as pd
import random
import numpy as np

# Specify the path and filename of the Excel file

df = pd.read_csv('data.csv.txt', header=None)

# Generate Random Samples of 100.
sample_index = random.sample(range(1, 900), 500)

def method1():
    A_temp = []
    b_temp = []

    alphas = []
    betas = []
    ks = []

    for k in range(1000):
        A_temp = []
        b_temp = []
        A = []
        b = []
        x = []
        for l in range(2):
            sample_index = random.sample(range(1, 999), 500)

            # Get average of datasets
            sample_english_average = sum([df[0][i] for i in sample_index])/500
            sample_hindi_average = sum([df[1][i] for i in sample_index])/500
            sample_sanskrit_average = sum([df[3][i] for i in sample_index])/500

            Ai = [sample_english_average, sample_hindi_average, 0.5]
            A_temp.append(Ai)
            b_temp.append(sample_sanskrit_average)
        A_temp.append([1,1,0])
        b_temp.append(1)

        A = np.array(A_temp)
        b = np.array(b_temp)

        x = np.linalg.solve(A, b)

        alphas.append(x[0])
        betas.append(x[1])
        ks.append(x[2])

    print(sum(alphas)/len(alphas))
    print(sum(betas)/len(betas))
    print(sum(ks)/len(ks))

def method2():
    alpha = random.random()
    beta = 1-alpha
    weights = [alpha, beta]
    k = random.random()*100
    bias = k
    i =0
    prev_error = 1
    for i in range(999):
        pred_val = alpha*df[0][i] + beta*df[1][i] + k
        actual_val = df[0][3]

        error = (pred_val-actual_val)**2

        grad_w = [(error*2*w)*(1/1000) for w in weights]
        grad_b = error*2 * (1/1000)

        learning_rate = 0.001
        weights = [weights[k] - learning_rate*grad_w[k] for k in range(len(weights)) ]
        bias = bias - grad_b
    print(alpha,beta,bias)

method2()









