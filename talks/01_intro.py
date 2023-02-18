# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# # Sieci neuronowe
#
# Wprowadzenie

# + [markdown] tags=[] slideshow={"slide_type": "slide"}
# ## Plan
#
# 1. Uczenie maszynowe
# 2. Prosta sieć neuronowa
# 3. Rozpoznawanie cyfr pisanych (MNIST)
# 4. Plan na następne talki

# + slideshow={"slide_type": "skip"} tags=[]
# %matplotlib inline

from ipywidgets import interactive
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import torch


# + slideshow={"slide_type": "skip"} tags=[]
def plt_setup():
    plt.subplots(figsize=(4,3))
    plt.grid(True)

def plot_func(*funcs):
    plt_setup()
    x = np.linspace(-2, 2, 200)
    
    for f in funcs:
        plt.plot(x, f(x))
        
    plt.show()


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# Spróbujmy odgadnąć parametry funkcji kwadratowej

# +
def quad(a, b, c):
    return lambda x: a * x**2 + b * x + c

def interactive_quad(f):
    return interactive(
        lambda a=1, b=1, c=0: f(a,b,c),
        a=(-2, 2, 0.1), b=(-2, 2, 0.1),  c=(-2, 2, 0.1),
    )

interactive_quad(lambda a,b,c: plot_func(quad(a,b,c)))

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# Wylosujmy jakieś parametry funkcji

# +
target_params = tuple(np.random.uniform(-2, 2, 3))
target = quad(*target_params)

plot_func(target)
target_params

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# Czy możemy ręcznie znaleźć parametry?
# -

interactive_quad(lambda a, b, c: plot_func(quad(a,b,c), target))

# Jak można zautomatyzować ten proces?

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ### Zbiór uczący
#
# Nasze obserwacje

# +
x = np.sort(np.random.uniform(-2, 2, 100))
y = target(x) + np.random.uniform(0, 0.2, x.shape[0])

plt_setup()
plt.scatter(x, y, s=3)
plt.show()


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ### Funkcja straty
#
# Jak daleko jesteśmy od celu?
# -

def mean_squared_error(target, prediction):
    return ((target - prediction) ** 2).mean()


# + slideshow={"slide_type": "skip"} tags=[]
def plot_func_with_loss(a,b,c):
    y_predicted = quad(a,b,c)(x)
    
    plt_setup()
    plt.scatter(x, y, s=3, c="C0")
    plt.plot(x, y_predicted, c="C1")
    plt.show()
    
    loss = mean_squared_error(y, y_predicted)
    print("loss =", loss)


# -

interactive_quad(plot_func_with_loss)


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ### Gradient funkcji straty
#
# W którą stronę iść?

# + tags=[] slideshow={"slide_type": "skip"}
def loss_line(predict):
    A = np.linspace(-2, 2, 50)    
    Y_predicted = predict(A)(np.broadcast_to(x, A.shape + x.shape).transpose())
    Y = np.broadcast_to(y, A.shape + y.shape).transpose()

    return A, ((Y - Y_predicted)**2).mean(axis=0)

def plot_loss(a, b, c):
    y_predicted = quad(a,b,c)(x)
    loss = mean_squared_error(y, y_predicted)
    
    print("loss =", loss)
    
    _, (a_ax, b_ax, c_ax) = plt.subplots(1, 3, figsize=(12,2), sharey=True)
    
    a_loss_line = loss_line(lambda a: quad(a, b, c))
    a_ax.plot(*a_loss_line, c="C0")
    a_ax.scatter(a, loss, c="C1")
    a_ax.set_xlabel("a")
    a_ax.set_ylabel("loss")
    a_ax.set_ylim([-1, 30])
    
    b_loss_line = loss_line(lambda b: quad(a, b, c))
    b_ax.plot(*b_loss_line, c="C0")
    b_ax.scatter(b, loss, c="C1")
    b_ax.set_xlabel("b")
        
    c_loss_line = loss_line(lambda c: quad(a, b, c))
    c_ax.plot(*c_loss_line, c="C0")
    c_ax.scatter(c, loss, c="C1")
    c_ax.set_xlabel("c")
        
    _, fit_ax = plt.subplots(figsize=(3,2))
    fit_ax.scatter(x, y, s=3, c="C0")
    fit_ax.plot(x, y_predicted, c="C1")
    fit_ax.set_xlabel("x")
    fit_ax.set_ylabel("y")
        
    plt.show()


# + tags=[]
interactive_quad(plot_loss)


# + tags=[] slideshow={"slide_type": "skip"}
def loss_with_gradient(a, b, c):
    a = torch.tensor(a, requires_grad=True)
    b = torch.tensor(b, requires_grad=True)
    c = torch.tensor(c, requires_grad=True)
    
    y_predicted = quad(a, b, c)(torch.from_numpy(x))
    
    loss = mean_squared_error(torch.from_numpy(y), y_predicted)
    loss.backward()
    
    return (
        y_predicted.detach().numpy(), 
        loss.item(),
        np.array([a.grad.item(), b.grad.item(), c.grad.item()])
    )

def plot_loss_gradient(a, b, c):
    y_predicted, loss, grad = loss_with_gradient(a, b, c)
    lr = (loss / grad ** 2).min()
    
    print("loss     =", loss)
    print("gradient =", tuple(grad))
    
    _, (a_ax, b_ax, c_ax) = plt.subplots(1, 3, figsize=(12,2), sharey=True)
    
    a_ax.axline([a, loss], [a+1, loss+grad[0]], c="C0")
    a_ax.quiver(a, loss, -lr * grad[0], 0, color="C1", angles='xy', scale_units='xy', scale=1)
    a_ax.scatter(a, loss, c="C1")
    a_ax.set_xlabel("a")
    a_ax.set_ylabel("loss")
    a_ax.set_xlim([-2, 2])
    c_ax.set_ylim([-1, 30])
    
    b_ax.axline([b, loss], [b+1, loss+grad[1]], c="C0")
    b_ax.quiver(b, loss, -lr * grad[1], 0, color="C1", angles='xy', scale_units='xy', scale=1)
    b_ax.scatter(b, loss, c="C1")
    b_ax.set_xlabel("b")
    b_ax.set_xlim([-2, 2])
        
    c_ax.axline([c, loss], [c+1, loss+grad[2]], c="C0")
    c_ax.quiver(c, loss, -lr * grad[2], 0, color="C1", angles='xy', scale_units='xy', scale=1)
    c_ax.scatter(c, loss, c="C1")
    c_ax.set_xlabel("c")
    c_ax.set_xlim([-2, 2])
    
        
    _, fit_ax = plt.subplots(figsize=(3,2))
    fit_ax.scatter(x, y, s=3, c="C0")
    fit_ax.plot(x, y_predicted, c="C1")
    fit_ax.set_xlabel("x")
    fit_ax.set_ylabel("y")
        
    plt.show()

# + tags=[]
interactive_quad(plot_loss_gradient)
# -

# ### Learning rate
#
# Jak szybko iść?


