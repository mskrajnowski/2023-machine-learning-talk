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
    display(loss)


# -

interactive_quad(plot_func_with_loss)

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ### Gradient straty
#
# W którą stronę iść?

# + tags=[] slideshow={"slide_type": "skip"}
from matplotlib import cm

def loss_grid(predict):
    A, B = np.meshgrid(
        np.linspace(-2, 2, 20),
        np.linspace(-2, 2, 20),
    )
    
    Y_predicted = predict(A, B)(np.broadcast_to(x, A.shape + x.shape).transpose(2, 0, 1))
    Y = np.broadcast_to(y, A.shape + y.shape).transpose(2, 0, 1)

    return A, B, ((Y - Y_predicted)**2).mean(axis=0)

def plot_loss():
    target_a, target_b, target_c = target_params
    
    fig, (ab_ax, ac_ax, bc_ax) = plt.subplots(1 ,3, figsize=(15,5), subplot_kw=dict(projection="3d"))
    
    ab_loss_grid = loss_grid(lambda a,b: quad(a,b,target_c))
    ab_ax.plot_surface(*ab_loss_grid, alpha=0.5, cmap=cm.coolwarm)
    ab_ax.contour(*ab_loss_grid, levels=30, cmap=cm.coolwarm)
    ab_ax.scatter(target_a, target_b, 0, c="blue")
    ab_ax.set_xlabel("a")
    ab_ax.set_ylabel("b")
    ab_ax.set_zlim(0, 30)
    ab_ax.set_title("loss(a,b)")
    
    ac_loss_grid = loss_grid(lambda a,c: quad(a,target_b,c))
    ac_ax.plot_surface(*ac_loss_grid, alpha=0.5, cmap=cm.coolwarm)
    ac_ax.contour(*ac_loss_grid, levels=30, cmap=cm.coolwarm)
    ac_ax.scatter(target_a, target_c, 0, c="blue")
    ac_ax.set_xlabel("a")
    ac_ax.set_ylabel("c")
    ac_ax.set_zlim(0, 30)
    ac_ax.set_title("loss(a,c)")
    
    bc_loss_grid = loss_grid(lambda b,c: quad(target_a,b,c))
    bc_ax.plot_surface(*bc_loss_grid, alpha=0.5, cmap=cm.coolwarm)
    bc_ax.contour(*bc_loss_grid, levels=30, cmap=cm.coolwarm)
    bc_ax.scatter(target_b, target_c, 0, c="blue")
    bc_ax.set_xlabel("b")
    bc_ax.set_ylabel("c")
    bc_ax.set_zlim(0, 30)
    bc_ax.set_title("loss(b,c)")

    plt.show()


# + tags=[]
plot_loss()

# +
from sympy import symbols, diff, simplify

def sym_mean_squared_error(target, prediction):
    target_values = list(target)
    predicted_values = list(prediction)
    
    squared_errors = [
        (target_values[i] - predicted_values[i])**2 
        for i in range(len(target))
    ]
    
    return sum(squared_errors) / len(squared_errors)


# -

a, b, c = symbols("a b c")

loss = sym_mean_squared_error(y, quad(a,b,c)(x))

dloss_a, dloss_b, dloss_c = diff(loss, a), diff(loss, b), diff(loss, c)
display(dloss_a, dloss_b, dloss_c)


# +
def gradient(a_val, b_val, c_val):    
    return (
        dloss_a.evalf(subs={a: a_val, b: b_val, c: c_val}),
        dloss_b.evalf(subs={a: a_val, b: b_val, c: c_val}),
        dloss_c.evalf(subs={a: a_val, b: b_val, c: c_val}),
    )
    
gradient(0.5, 0.75, 1)
# +
import torch

def plot_func_with_loss_and_gradient(a,b,c):  
    ta = torch.tensor(a, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    tc = torch.tensor(c, requires_grad=True)
    
    ty = torch.from_numpy(y)
    ty_predicted = quad(ta,tb,tc)(torch.from_numpy(x))
    y_predicted = ty_predicted.detach().numpy()
    
    plt_setup()
    plt.scatter(x, y, s=3, c=plot_colors[0])
    plt.plot(x, y_predicted, c=plot_colors[1])
    plt.show()
    
    tloss = mean_squared_error(ty, ty_predicted)
    display(tloss.item())
    
    tloss.backward()
    display((ta.grad.item(), tb.grad.item(), tc.grad.item()))


# -

interactive_quad(plot_func_with_loss_and_gradient)

# ### Learning rate
#
# Jak szybko iść?
