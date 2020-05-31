import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(12345)

x1 = np.arange(0, 5, 0.2)

a = 0.7
b = 1.3
y1 = a*x1 + b + np.random.randn(x1.size)
x2 = np.arange(-1, 1, 0.1)
a3 = 0.9
a2 = 0.8
a1 = -0.7
a0 = 1.2

y2 = a3*x2**3 + a2*x2**2 + a1*x2**1 + a0*x2**0 + np.random.randn(x2.size)*0.05

x3 = np.arange(0, 2*np.pi, 0.3)
y3 = 2*np.sin(x3) + np.random.rand(x3.size)*0.2





x1 = x1.reshape(len(y1),-1)

reg1 = LinearRegression().fit(x1,y1)
y1_pred = reg1.predict(x1)
print(reg1.coef_)
print(reg1.intercept_)


#plt.show()

x2 = x2.reshape(len(y2),-1)
x2 = np.vstack(x2)
# print(x2)
# print(y2)
polynomial_features = PolynomialFeatures(degree=3)
x2_poly = polynomial_features.fit_transform(x2)
reg2 = LinearRegression().fit(x2_poly,y2)
print(reg2.coef_)
print(reg2.intercept_)
y2_poly_pred = reg2.predict(x2_poly)



x3 = x3.reshape(len(y3), -1)
x3_sine = x3
x3_sine = np.hstack([np.sin(x3)])
reg3  = LinearRegression().fit(x3_sine,y3)
print(reg3.coef_)
y3_sine_pred = reg3.predict(x3_sine)

plt.figure(dpi=100)
plt.subplot(3, 1, 1)
plt.plot(x1, y1, "r*")
plt.plot(x1,y1_pred, color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(x2, y2, "g*")
plt.plot(x2, y2_poly_pred, color= 'green')
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(x3, y3, "b*")
plt.plot(x3, y3_sine_pred, color='blue')
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.tight_layout()
plt.show()






