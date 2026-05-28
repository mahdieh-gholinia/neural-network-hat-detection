import numpy as np
import matplotlib.pyplot as plt


def make_data(n):
    x = np.random.uniform(-10, 20, n)
    y = np.random.uniform(-10, 20, n)
    z = np.random.uniform(-10, 20, n)

    hat = (x - 3)**2 + (y - 5)**2 + z
    # بالای کلاه = 1 ، پایین یا روی کلاه = 0 
    lbl = (hat> 8).astype(int)
    return np.vstack([x, y, z]).T,lbl.reshape(-1,1)


X_train, y_train = make_data(10000)
X_test,  y_test  = make_data(2000)


def sig(x):
    return 1 / (1 + np.exp(-x))

def sig_der(x):
    s = sig(x)
    return s * (1 - s)

# ساخت وزن‌ها و بایوس
np.random.seed(1)
W1 = np.random.randn(3,10)*0.1
b1 = np.zeros((1,10))

W2 = np.random.randn(10,10)*0.1
b2 = np.zeros((1,10))

W3 = np.random.randn(10,10)*0.1
b3 = np.zeros((1,10))

W4 = np.random.randn(10,1)*0.1
b4 = np.zeros((1,1))

lr = 0.0001


loss_history = []
acc_history = [] 

# آموزش شبکه
for ep in range(100):

    # Forward
    z1 = np.dot(X_train, W1) + b1
    a1 = sig(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sig(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = sig(z3)

    z4 = np.dot(a3, W4) + b4
    y_p = sig(z4)

    # محاسبه خطا
    loss = np.mean((y_train - y_p)**2)
    loss_history.append(loss)  

   

    # Backprop
    d4 = (y_p - y_train) * sig_der(z4)
    dW4 = np.dot(a3.T, d4)
    db4 = np.sum(d4, axis=0, keepdims=True)

    d3 = np.dot(d4, W4.T) * sig_der(z3)
    dW3 = np.dot(a2.T, d3)
    db3 = np.sum(d3, axis=0, keepdims=True)

    d2 = np.dot(d3, W3.T) * sig_der(z2)
    dW2 = np.dot(a1.T, d2)
    db2 = np.sum(d2, axis=0, keepdims=True)

    d1 = np.dot(d2, W2.T) * sig_der(z1)
    dW1 = np.dot(X_train.T, d1)
    db1 = np.sum(d1, axis=0, keepdims=True)

    # آپدیت وزن ها
    W4 -= lr * dW4
    b4 -= lr * db4

    W3 -= lr * dW3
    b3 -= lr * db3

    W2 -= lr * dW2
    b2 -= lr * db2

    W1 -= lr * dW1
    b1 -= lr * db1

    # چاپ هر 10 اپوک
    if ep % 10 == 0:
        print("epoch:", ep, "loss:", loss)

#تست شبکه
z1_t = np.dot(X_test, W1) + b1
a1_t = sig(z1_t)

z2_t = np.dot(a1_t, W2) + b2
a2_t = sig(z2_t)

z3_t = np.dot(a2_t, W3) + b3
a3_t = sig(z3_t)

z4_t = np.dot(a3_t, W4) + b4
y_test_p = sig(z4_t)

pred = (y_test_p > 0.5).astype(int)
acc = np.mean(pred == y_test)
acc_history.append(acc)
    
print("epoch: 100 loss:", loss, "accuracy:", acc)
print("accuracy:",acc)


# نمودار loss
plt.figure()
plt.plot(range(1, 101), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.grid(True)
plt.ylim(min(loss_history)*0.999, max(loss_history)*1.001)
plt.show()
