import numpy as np

def our_linear_regression_matrix(X,y):

    lin1 = np.linalg.inv(np.dot(X.T,X)) # ze wzoru odwrócony iloczyn(inv) macierzy transponowanej(.T) i tej samej macierzy
    lin2 = np.dot(X.T, y) # macierz transponowana X i macierz y
    lin3 = np.dot(lin1, lin2) # wynik lin2 i lin3, dot zwraca iloczyn tablic, które w tym przypadku są macierzami
    return lin3
def our_linear_regression_iterative(X, y, learning_rate = 0.01, num_iters = 11000):

    c = float(len(y))
    d = np.zeros(np.size(X,1))
    for i in range(num_iters):
        e = np.dot(X,d)
        d = d - (2/c)*learning_rate*(X.T.dot((e - y)))
    return d

np.random.seed(12345)

x1 = np.arange(0, 5, 0.2)
a = 0.7
b = 1.3
y1 = a*x1 + b + np.random.randn(x1.size)
x1 = np.c_[np.ones(len(y1)),x1]    # ones tworzy macierz jednostkową o długości y1 bo przekaywana tablica przy inv musi byc min 2D
print(our_linear_regression_matrix(x1,y1)) # względem osi x
print(our_linear_regression_iterative(x1,y1))
x2 = np.arange(-1, 1, 0.1)
a3 = 0.9
a2 = 0.8
a1 = -0.7
a0 = 1.2
y2 = a3*x2**3 + a2*x2**2 + a1*x2**1 + a0*x2**0 + np.random.randn(x2.size)*0.05
x2 = x2.reshape(len(y2),-1) # zmiana kształtu macierzy, teraz ma jedną kolumne oraz ilość wierszy odpowiadającą ilości kolumn macierzy y2
x2 = np.hstack([x2**1, x2**2,x2**3]) # hstack łączy 3 macierze powstaje wtedy macierz 3 kolumny a jej elementami są element z x2 do potęg
x2 = np.c_[np.ones(len(y2)),x2]
print(our_linear_regression_matrix(x2,y2))
print(our_linear_regression_iterative(x2,y2))
x3 = np.arange(0, 2*np.pi, 0.3)
y3 = 2*np.sin(x3) + np.random.rand(x3.size)*0.2
x3 = np.hstack([np.sin(x3)])
x3 = x3.reshape(len(y3),-1)
print(our_linear_regression_matrix(x3,y3))
print(our_linear_regression_iterative(x3,y3))