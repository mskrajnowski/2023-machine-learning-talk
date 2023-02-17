# # Sieci neuronowe
#
# Wprowadzenie

# ## Plan
#
# 1. Uczenie maszynowe
# 2. Prosta sieć neuronowa
# 3. Rozpoznawanie cyfr pisanych (MNIST)
# 4. Plan na następne talki

# +
# %matplotlib inline

from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# +
plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def plt_setup():
    plt.subplots(figsize=(4,3))
    plt.grid(True)

def plot_func(*funcs):
    plt_setup()
    x = np.linspace(-2, 2, 200)
    
    for f in funcs:
        plt.plot(x, f(x))
        
    plt.show()


# -

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
# -

# Wylosujmy jakieś parametry funkcji

# +
target_params = tuple(np.random.uniform(-2, 2, 3))
target = quad(*target_params)

plot_func(target)
target_params
# -

# Czy możemy ręcznie znaleźć parametry?

interactive_quad(lambda a, b, c: plot_func(quad(a,b,c), target))

# Jak można zautomatyzować ten proces?

# ### Zbiór uczący
#
# Nasze obserwacje

# +
x = np.sort(np.random.uniform(-2, 2, 100))
y = target(x) + np.random.uniform(0, 0.1, x.shape[0])

plt_setup()
plt.scatter(x, y, s=3)
plt.show()


# -

# ### Funkcja straty
#
# Jak daleko jesteśmy od celu?

def mean_squared_error(target, prediction):
    return ((target - prediction) ** 2).mean()


def plot_func_with_loss(a,b,c):
    y_predicted = quad(a,b,c)(x)
    
    plt_setup()
    plt.scatter(x, y, s=3, c=plot_colors[0])
    plt.plot(x, y_predicted, c=plot_colors[1])
    plt.show()
    
    loss = mean_squared_error(y, y_predicted)
    display(loss)


interactive_quad(plot_func_with_loss)

# ### Stochastic Gradient Descent (SGD)
#
# W którą stronę iść?

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
# -



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


interactive_quad(plot_func_with_loss_and_gradient)

# ### Learning rate
#
# Jak szybko iść?
