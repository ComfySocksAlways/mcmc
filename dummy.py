import numpy as np

def tranformsl(A:np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray):
    for i in range(X.shape[0]):
        for j in range(A.shape[1]):
            temp = 0
            for k in range(A.shape[0]):
                temp += X[i,k] * A[k,j]
            Y[i,j] = temp + b[0,j]



if __name__ == "__main__":
    N = 20
    A = np.random.random((5,5)).astype(np.float32)
    X = np.random.random((N,5)).astype(np.float32)
    b = np.random.random((1,5)).astype(np.float32)
    Y = np.zeros_like(X).astype(np.float32)
    tranformsl(A,b,X,Y)