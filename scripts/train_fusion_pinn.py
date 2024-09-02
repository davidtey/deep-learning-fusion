import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

import mlflow
import mlflow.pytorch

from tqdm.auto import tqdm

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Train Fusion PINN")

# Compulsory arguments
parser.add_argument("-n", "--experiment_name", type=str, default="fusion_pinn", help="Name of the experiment")
parser.add_argument("-l", "--layers", type=int, default=4, help="Number of layers")
parser.add_argument("-w", "--width", type=int, default=64, help="Width of the network")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("-e", "--epochs", type=int, default=10000, help="Number of epochs")
parser.add_argument("-ei", "--eval_interval", type=int, default=100, help="Evaluation interval")
parser.add_argument("-intbs", "--interior_batch_size", type=int, default=2048, help="Interior batch size")
parser.add_argument("-inibs", "--initial_batch_size", type=int, default=2048, help="Initial batch size")
parser.add_argument("-bbs", "--boundary_batch_size", type=int, default=2048, help="Boundary batch size")
parser.add_argument("--x_min", type=float, default=0.0, help="Minimum x value")
parser.add_argument("--x_max", type=float, default=1.0, help="Maximum x value")
parser.add_argument("--t_min", type=float, default=0.0, help="Minimum t value")
parser.add_argument("--t_max", type=float, default=1.0, help="Maximum t value")
parser.add_argument("-sint", "--sample_interval", type=int, default=np.inf, help="Sample interval")
parser.add_argument("-intlf", "--interior_loss_function", type=str, default="l2", help="Interior loss function")
parser.add_argument("-inilf", "--initial_loss_function", type=str, default="l2", help="Initial loss function")
parser.add_argument("-blf", "--boundary_loss_function", type=str, default="l2", help="Boundary loss function")
parser.add_argument("-fpm", "--force_positive_method", type=str, default="exp", help="Force positive method")
parser.add_argument("-o", "--optimizer", type=str, default="adam", help="Optimizer")
parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use")
parser.add_argument("-p", "--precision", type=str, default="float64", help="Precision")

# Optional sine initialisation
parser.add_argument("-sinit", "--sine_initialisation", type=bool, default=False, help="Sine initialisation")
parser.add_argument("-sinit_epochs", "--sine_initialisation_epochs", type=int, default=100000, help="Sine initialisation epochs")
parser.add_argument("-sinit_lr", "--sine_initialisation_learning_rate", type=float, default=1e-3, help="Sine initialisation learning rate")
parser.add_argument("-sinit_ei", "--sine_initialisation_eval_interval", type=int, default=5000, help="Sine initialisation evaluation interval")

# Optional multistage training
parser.add_argument("-mst", "--multistage_training", type=bool, default=False, help="Multistage training")

# Optional separate modeling
parser.add_argument("-sm", "--separate_models", type=bool, default=False, help="Separate models")

# Outputs
parser.add_argument("-po", "--plot_outputs", type=bool, default=True, help="Plot outputs")
parser.add_argument("-pv", "--plot_vars", type=str, default="n, n_t, n_x, n_xx, u, u_t, u_x, u_xx, ϵ, ϵ_t, ϵ_x", help="Variables to plot")

args = parser.parse_args()

args.sample_interval = np.inf if args.sample_interval == 0 else args.sample_interval
args.plot_vars = args.plot_vars.split(", ")

# physical parameters
l_0 = 1.0
κ = 1.0
α = 6.0
D_c = 0.78
C_χ = 0.95
a_u = 1.0
μ_c = 0.78
β = 0.1
Λ = 4000.0
ϵ_c = 6.25

g_i = 5.1
ϵ_i = 0.002

physical_params = {
    'l_0': l_0,
    'κ': κ,
    'α': α,
    'D_c': D_c,
    'C_χ': C_χ,
    'a_u': a_u,
    'μ_c': μ_c,
    'β': β,
    'Λ': Λ,
    'ϵ_c': ϵ_c,
    'g_i': g_i,
    'ϵ_i': ϵ_i
}

# Check arguments valid
assert args.layers > 0
assert args.width > 0
assert args.learning_rate > 0
assert args.epochs > 0
assert 0 < args.eval_interval <= args.epochs
assert args.interior_batch_size > 0
assert args.initial_batch_size > 0
assert args.boundary_batch_size > 0
assert args.x_min < args.x_max
assert args.t_min < args.t_max
assert args.sample_interval >= 0
assert args.interior_loss_function in ["l1", "l2", "linf"]
assert args.initial_loss_function in ["l1", "l2", "linf"]
assert args.boundary_loss_function in ["l1", "l2", "linf"]
assert args.force_positive_method in ["exp", "abs", "square", "softplus"]
assert args.optimizer in ["adam", "adamw", "sgd"]
assert args.device in ["cuda", "cpu"]
assert args.precision in ["float32", "float64"]
assert args.sine_initialisation_epochs > 0
assert args.sine_initialisation_learning_rate > 0
assert 0 < args.sine_initialisation_eval_interval <= args.sine_initialisation_epochs
assert set(args.plot_vars).issubset(["n", "n_t", "n_x", "n_xx", "u", "u_t", "u_x", "u_xx", "ϵ", "ϵ_t", "ϵ_x"])

class Exponentialϵ(nn.Module):
    def __init__(self):
        super(Exponentialϵ, self).__init__()
        
    def forward(self, x):
        return torch.exp(x)
    
class Absoluteϵ(nn.Module):
    def __init__(self):
        super(Absoluteϵ, self).__init__()
        
    def forward(self, x):
        return torch.abs(x + 1e-6)

class Squareϵ(nn.Module):
    def __init__(self):
        super(Squareϵ, self).__init__()
        
    def forward(self, x):
        return (x + 1e-6) ** 2

class ForcePositiveϵ(nn.Module):
    def __init__(self, args):
        super(ForcePositiveϵ, self).__init__()
        self.n = nn.Sequential(nn.Linear(args.width, 1))
        self.u = nn.Sequential(nn.Linear(args.width, 1))

        if args.force_positive_method == "exp":
            self.ε = nn.Sequential(nn.Linear(args.width, 1), Exponentialε())
        elif args.force_positive_method == "abs":
            self.ε = nn.Sequential(nn.Linear(args.width, 1), Absoluteε())
        elif args.force_positive_method == "square":
            self.ε = nn.Sequential(nn.Linear(args.width, 1), Squareε())
        elif args.force_positive_method == "softplus":
            self.ϵ = nn.Sequential(nn.Linear(args.width, 1), nn.Softplus())
            
        self.n.to(args.device)
        self.u.to(args.device)
        self.ε.to(args.device)
            
    def forward(self, x):
        return torch.hstack((self.n(x), self.u(x), self.ε(x)))
    
    def parameters(self):
        return list(self.n.parameters()) + list(self.u.parameters()) + list(self.ε.parameters())
    
class SeparatedPINN(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # model for n
        self.modules_n = [nn.BatchNorm1d(2), nn.Linear(2, args.width), nn.GELU()]
        for _ in range(args.layers - 1):
            self.modules_n.append(nn.Linear(args.width, args.width))
            self.modules_n.append(nn.GELU())
        
        self.model_n = nn.Sequential(*self.modules_n)
        self.model_n.to(args.device)
        
        # model for u
        self.modules_u = [nn.BatchNorm1d(2), nn.Linear(2, args.width), nn.GELU()]
        for _ in range(args.layers - 1):
            self.modules_u.append(nn.Linear(args.width, args.width))
            self.modules_u.append(nn.GELU())
        
        self.model_u = nn.Sequential(*self.modules_u)
        self.model_u.to(args.device)
        
        # model for ϵ
        self.modules_ϵ = [nn.BatchNorm1d(2), nn.Linear(2, args.width), nn.GELU()]
        for _ in range(args.layers - 1):
            self.modules_ϵ.append(nn.Linear(args.width, args.width))
            self.modules_ϵ.append(nn.GELU())
        
        self.model_ϵ = nn.Sequential(*self.modules_ϵ)
        self.model_ϵ.to(args.device)
        
        self.output_layer = ForcePositiveϵ(args)
        
    def forward(self, x):
        return self.output_layer(torch.hstack((self.model_n(x), self.model_u(x), self.model_ϵ(x))))
    
    def parameters(self):
        return list(self.model_n.parameters()) + list(self.model_u.parameters()) + list(self.model_ϵ.parameters()) + list(self.output_layer.parameters())

class CombinedPINN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.modules = [nn.BatchNorm1d(2), nn.Linear(2, args.width), nn.GELU()]
        for i in range(args.layers - 1):
            self.modules.append(nn.Linear(args.width, args.width))
            self.modules.append(nn.GELU())
        
        self.modules.append(ForcePositiveϵ(args))
        self.model = nn.Sequential(*self.modules)
        self.model.to(args.device)
        
    def forward(self, x):
        return self.model(x)
    
    def parameters(self):
        return list(self.model.parameters())
    
class PINN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
        if args.separate_models:
            self.model = SeparatedPINN(args)
        else:
            self.model = CombinedPINN(args)
            
        self.parameters = self.model.parameters()
        self.num_parameters = sum(p.numel() for p in self.parameters)
        
        if args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters, lr=args.learning_rate)
        elif args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.parameters, lr=args.learning_rate)
        elif args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters, lr=args.learning_rate)
        
        self.epoch = 0
        
        t = np.linspace(args.t_min, args.t_max, 100)
        x = np.linspace(args.x_min, args.x_max, 100)

        self.eval_t, self.eval_x = np.meshgrid(t, x)
        self.eval_t = torch.Tensor(self.eval_t).reshape(-1, 1).to(args.device)
        self.eval_x = torch.Tensor(self.eval_x).reshape(-1, 1).to(args.device)
        
        self.eval_t.requires_grad_()
        self.eval_x.requires_grad_()
        
        eval_t_interior, eval_x_interior = np.meshgrid(t[1:-1], x[1:-1])
        eval_t_interior = torch.Tensor(eval_t_interior).reshape(-1, 1).to(args.device)
        eval_x_interior = torch.Tensor(eval_x_interior).reshape(-1, 1).to(args.device)
        
        self.eval_X_interior = torch.hstack((eval_t_interior, eval_x_interior))
        self.eval_X_interior.requires_grad_()
        
        eval_t_initial = torch.zeros_like(self.eval_x)
        self.eval_X_initial = torch.hstack((eval_t_initial, self.eval_x))
        self.eval_X_initial.requires_grad_()
        
        eval_t_boundary = torch.Tensor(np.vstack([t, t])).reshape(-1, 1).to(args.device)
        eval_x_boundary = torch.Tensor(np.vstack([x[0] * np.ones_like(t), x[-1] * np.ones_like(t)])).reshape(-1, 1).to(args.device)
        
        self.eval_X_boundary = torch.hstack((eval_t_boundary, eval_x_boundary))
        self.eval_X_boundary.requires_grad_()
        
        self.plot_t = self.eval_t.view(100, 100).detach().cpu().numpy()
        self.plot_x = self.eval_x.view(100, 100).detach().cpu().numpy()
        
        self.plot_files = {}
        
        if args.plot_outputs:
            for varname in args.plot_vars:
                self.plot_files[varname] = []
    
    def __call__(self, X):
        return self.model(X)
    
    # residual losses
    def residual_n_t_loss(self, x, n_t, n_x, n_xx, ϵ, l):    
        intermediate = l**2 * ϵ * n_x / α
        intermediate_x = grad(intermediate, x, grad_outputs=torch.ones_like(intermediate), retain_graph=True, create_graph=True)[0]
        loss = n_t - intermediate_x - D_c * n_xx
        
        if self.args.interior_loss_function == 'l1':
            return torch.mean(torch.abs(loss))
        elif self.args.interior_loss_function == 'l2':
            return torch.mean(loss ** 2)
        elif self.args.interior_loss_function == 'linf':
            return torch.max(torch.abs(loss))
        
    def residual_u_t_loss(self, x, ϵ, n_x, u_t, u_xx, l, w):    
        intermediate = (l**2 * ϵ / α - w) * n_x
        intermediate_x = grad(intermediate, x, grad_outputs=torch.ones_like(intermediate), retain_graph=True, create_graph=True)[0]
        
        loss = u_t - intermediate_x - w * u_xx - μ_c * u_xx
        
        if self.args.interior_loss_function == 'l1':
            return torch.mean(torch.abs(loss))
        elif self.args.interior_loss_function == 'l2':
            return torch.mean(loss ** 2)
        elif self.args.interior_loss_function == 'linf':
            return torch.max(torch.abs(loss))
    
    def residual_ϵ_t_loss(self, x, ϵ, n_x, u_x, ϵ_t, ϵ_x, l, w):
        intermediate = l**2 * torch.sqrt(ϵ) * ϵ_x
        intermediate_x = grad(intermediate, x, grad_outputs=torch.ones_like(intermediate), retain_graph=True, create_graph=True)[0]
        
        loss = ϵ_t - β * intermediate_x - Λ * (w * (n_x - u_x)**2 - ϵ**(3/2) / ϵ_c**0.5 + ϵ)
        
        if self.args.interior_loss_function == 'l1':
            return torch.mean(torch.abs(loss))
        elif self.args.interior_loss_function == 'l2':
            return torch.mean(loss ** 2)
        elif self.args.interior_loss_function == 'linf':
            return torch.max(torch.abs(loss))
    
    # initial conditions
    def n_initial_cond(self, t, x):
        return -g_i * x

    def u_initial_cond(self, t, x):
        return torch.zeros(x.shape, device=x.device.type)

    def ϵ_initial_cond(self, t, x):
        return torch.full(x.shape, ϵ_i, device=x.device.type)
    
    # initial loss
    def n_initial_loss(self, t, x, n_pred):
        if self.args.initial_loss_function == 'l1':
            return torch.mean(torch.abs(n_pred - self.n_initial_cond(t, x)))
        elif self.args.initial_loss_function == 'l2':
            return torch.mean((n_pred - self.n_initial_cond(t, x))**2)
        elif self.args.initial_loss_function == 'linf':
            return torch.max(torch.abs(n_pred - self.n_initial_cond(t, x)))
        
    def u_initial_loss(self, t, x, u_pred):
        if self.args.initial_loss_function == 'l1':
            return torch.mean(torch.abs(u_pred - self.u_initial_cond(t, x)))
        elif self.args.initial_loss_function == 'l2':
            return torch.mean((u_pred - self.u_initial_cond(t, x))**2)
        elif self.args.initial_loss_function == 'linf':
            return torch.max(torch.abs(u_pred - self.u_initial_cond(t, x)))
    
    def ϵ_initial_loss(self, t, x, ϵ_pred):
        if self.args.initial_loss_function == 'l1':
            return torch.mean(torch.abs(ϵ_pred - self.ϵ_initial_cond(t, x)))
        elif self.args.initial_loss_function == 'l2':
            return torch.mean((ϵ_pred - self.ϵ_initial_cond(t, x))**2)
        elif self.args.initial_loss_function == 'linf':
            return torch.max(torch.abs(ϵ_pred - self.ϵ_initial_cond(t, x)))
    
    # boundary conditions
    def n_boundary_cond(self, t, x):
        out = torch.full(x.shape, -g_i, device=x.device.type)
        out = out * x
        
        return out

    def u_boundary_cond(self, t, x):
        return torch.zeros(x.shape, device=x.device.type)

    def ϵ_x_boundary_cond(self, t, x):
        return torch.zeros(x.shape, device=x.device.type)
    
    def n_boundary_loss(self, t, x, n_pred):
        if self.args.boundary_loss_function == 'l1':
            return torch.mean(torch.abs(n_pred - self.n_boundary_cond(t, x)))
        elif self.args.boundary_loss_function == 'l2':
            return torch.mean((n_pred - self.n_boundary_cond(t, x))**2)
        elif self.args.boundary_loss_function == 'linf':
            return torch.max(torch.abs(n_pred - self.n_boundary_cond(t, x)))
        
    def u_boundary_loss(self, t, x, u_pred):
        if self.args.boundary_loss_function == 'l1':
            return torch.mean(torch.abs(u_pred - self.u_boundary_cond(t, x)))
        elif self.args.boundary_loss_function == 'l2':
            return torch.mean((u_pred - self.u_boundary_cond(t, x))**2)
        elif self.args.boundary_loss_function == 'linf':
            return torch.max(torch.abs(u_pred - self.u_boundary_cond(t, x)))
    
    def ϵ_x_boundary_loss(self, t, x, ϵ_x_pred):
        if self.args.boundary_loss_function == 'l1':
            return torch.mean(torch.abs(ϵ_x_pred - self.ϵ_x_boundary_cond(t, x)))
        elif self.args.boundary_loss_function == 'l2':
            return torch.mean((ϵ_x_pred - self.ϵ_x_boundary_cond(t, x))**2)
        elif self.args.boundary_loss_function == 'linf':
            return torch.max(torch.abs(ϵ_x_pred - self.ϵ_x_boundary_cond(t, x)))
    
    # sample points
    def sample_interior_points(self, percent=1):
        t = torch.empty((self.args.interior_batch_size, 1), device=self.args.device).uniform_(self.args.t_min, percent * self.args.t_max)
        x1 = torch.empty((self.args.interior_batch_size//2, 1), device=self.args.device).uniform_(self.args.x_min, self.args.x_min + percent * (self.args.x_max - self.args.x_min))
        x2 = torch.empty((self.args.interior_batch_size//2, 1), device=self.args.device).uniform_(self.args.x_min + (1 - percent) * (self.args.x_max - self.args.x_min), self.args.x_max)
        x = torch.vstack((x1, x2))
        X_interior = torch.cat((t, x), 1)
        X_interior.requires_grad_()
        
        return X_interior
    
    def sample_initial_points(self):
        t = torch.zeros(self.args.initial_batch_size, 1, device=self.args.device)
        x = torch.empty((self.args.initial_batch_size, 1), device=self.args.device).uniform_(self.args.x_min, self.args.x_max)
        X_initial = torch.cat((t, x), 1)
        X_initial.requires_grad_()
        
        return X_initial
    
    def sample_boundary_points(self):
        options = torch.tensor([self.args.x_min, self.args.x_max], device=self.args.device)
        
        t = torch.empty((self.args.boundary_batch_size, 1), device=self.args.device).uniform_(self.args.t_min, self.args.t_max)
        x = options[torch.randint(0, 2, (self.args.boundary_batch_size, 1), device=self.args.device)]
        X_boundary = torch.cat((t, x), 1)
        X_boundary.requires_grad_()
        
        return X_boundary
    
    # plotting
    def plot_outputs(self):
        Y = self(torch.hstack((self.eval_t, self.eval_x)))

        n = Y[:, 0].view(-1, 1)
        u = Y[:, 1].view(-1, 1)
        ϵ = Y[:, 2].view(-1, 1)
        
        n_x = grad(n, self.eval_x, grad_outputs=torch.ones_like(n), retain_graph=True, create_graph=True)[0]
        n_t = grad(n, self.eval_t, grad_outputs=torch.ones_like(n), retain_graph=True)[0]
        
        n_xx = grad(n_x, self.eval_x, grad_outputs=torch.ones_like(n_x), retain_graph=True)[0]
        
        u_x = grad(u, self.eval_x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_t = grad(u, self.eval_t, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
        
        u_xx = grad(u_x, self.eval_x, grad_outputs=torch.ones_like(u_x), retain_graph=True)[0]
        
        ϵ_x = grad(ϵ, self.eval_x, grad_outputs=torch.ones_like(ϵ), retain_graph=True)[0]
        ϵ_t = grad(ϵ, self.eval_t, grad_outputs=torch.ones_like(ϵ), retain_graph=True)[0]
        
        n = n.view(100, 100).detach().cpu().numpy()
        n_x = n_x.view(100, 100).detach().cpu().numpy()
        n_t = n_t.view(100, 100).detach().cpu().numpy()
        n_xx = n_xx.view(100, 100).detach().cpu().numpy()
        
        u = u.view(100, 100).detach().cpu().numpy()
        u_x = u_x.view(100, 100).detach().cpu().numpy()
        u_t = u_t.view(100, 100).detach().cpu().numpy()
        u_xx = u_xx.view(100, 100).detach().cpu().numpy()
        
        ϵ = ϵ.view(100, 100).detach().cpu().numpy()
        ϵ_x = ϵ_x.view(100, 100).detach().cpu().numpy()
        ϵ_t = ϵ_t.view(100, 100).detach().cpu().numpy()
        
        if 'n' in self.args.plot_vars:
            self.plot_var(n, 'n')
        
        if 'n_x' in self.args.plot_vars:
            self.plot_var(n_x, 'n_x')
        
        if 'n_t' in self.args.plot_vars:
            self.plot_var(n_t, 'n_t')
        
        if 'n_xx' in self.args.plot_vars:
            self.plot_var(n_xx, 'n_xx')
        
        if 'u' in self.args.plot_vars:
            self.plot_var(u, 'u')
        
        if 'u_x' in self.args.plot_vars:
            self.plot_var(u_x, 'u_x')
        
        if 'u_t' in self.args.plot_vars:
            self.plot_var(u_t, 'u_t')
        
        if 'u_xx' in self.args.plot_vars:
            self.plot_var(u_xx, 'u_xx')
        
        if 'ϵ' in self.args.plot_vars:
            self.plot_var(ϵ, 'ϵ')
        
        if 'ϵ_x' in self.args.plot_vars:
            self.plot_var(ϵ_x, 'ϵ_x')
        
        if 'ϵ_t' in self.args.plot_vars:
            self.plot_var(ϵ_t, 'ϵ_t')
    
    def plot_var(self, var, varname):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.plot_t, self.plot_x, var, cmap='viridis')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel(varname)
        plt.title(f'Model output {varname}, epoch {self.epoch}')
        filename = f"plots/{self.args.experiment_name}/plot_{varname}_epoch_{self.epoch}.png"
        self.plot_files[varname].append(filename)
        fig.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()
    
    def save_gif(self):
        for varname in self.args.plot_vars:
            gif_filename = f"plots/{self.args.experiment_name}/plot_{varname}.gif"
            with imageio.get_writer(gif_filename, mode='I') as writer:
                for filename in self.plot_files[varname]:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                
            mlflow.log_artifact(gif_filename)
    
    # sine initialisation
    def initialise_starting_surface(self):
        T_init, X_init, n_init, u_init, ϵ_init = self.starting_surface()
        T_init.requires_grad_()
        X_init.requires_grad_()
        n_init.requires_grad_()
        u_init.requires_grad_()
        ϵ_init.requires_grad_()
        
        for i in tqdm(range(self.args.sine_initialisation_epochs), disable=True):
            self.optimizer.zero_grad()
            Y = self(torch.hstack((T_init, X_init)))
            
            n_pred = Y[:, 0].reshape(-1, 1)
            u_pred = Y[:, 1].reshape(-1, 1)
            ϵ_pred = Y[:, 2].reshape(-1, 1)
            
            mse = nn.MSELoss()
            
            loss_n = mse(n_pred, n_init)
            loss_u = mse(u_pred, u_init)
            loss_ϵ = mse(ϵ_pred, ϵ_init)
            
            loss = (loss_n + loss_u + loss_ϵ)/3
            
            loss.backward()
            self.optimizer.step()
            
            if i % self.args.sine_initialisation_eval_interval == 0:
                print(f"initialisation epoch: {i}, loss: {loss.item():,.4e}")
            
        print(f"initialisation complete, initial surface loss: {loss.item():,.4e}")
    
    def starting_surface(self):
        t = torch.linspace(self.args.t_min, self.args.t_max, 100, device=self.args.device)
        x = torch.linspace(self.args.x_min, self.args.x_max, 100, device=self.args.device)
        
        x_range = self.args.x_max - self.args.x_min
        t_range = self.args.t_max - self.args.t_min
        
        T, X = torch.meshgrid(t, x)
        n = torch.sin((7.5+torch.rand(1, device=self.args.device))*torch.pi*X/x_range + torch.rand(1, device=self.args.device)) * torch.sin((7.5+torch.rand(1, device=self.args.device))*torch.pi*T/t_range + torch.rand(1, device=self.args.device))
        u = torch.sin((7.5+torch.rand(1, device=self.args.device))*torch.pi*X/x_range + torch.rand(1, device=self.args.device)) * torch.sin((7.5+torch.rand(1, device=self.args.device))*torch.pi*T/t_range + torch.rand(1, device=self.args.device))
        ϵ = 1 + 1e-2 + torch.sin((7.5+torch.rand(1, device=self.args.device))*torch.pi*X/x_range + torch.rand(1, device=self.args.device)) * torch.sin((7.5+torch.rand(1, device=self.args.device))*torch.pi*T/t_range + torch.rand(1, device=self.args.device))
        
        return T.reshape(-1, 1), X.reshape(-1, 1), n.reshape(-1, 1), u.reshape(-1, 1), ϵ.reshape(-1, 1)
    
    # forward pass
    def forward(self, X_interior, X_initial, X_boundary):
        # X shape: (batch_size, 2), where 2nd dimension is [t, x]
        # Y shape: (batch_size, 3), where 2nd dimension is [n, u, ϵ]
        
        t_interior = X_interior[:, 0].reshape(-1, 1)
        x_interior = X_interior[:, 1].reshape(-1, 1)
        t_initial = X_initial[:, 0].reshape(-1, 1)
        x_initial = X_initial[:, 1].reshape(-1, 1)
        t_boundary = X_boundary[:, 0].reshape(-1, 1)
        x_boundary = X_boundary[:, 1].reshape(-1, 1)
        
        # forward pass
        Y_interior = self.model(torch.hstack((t_interior, x_interior)))
        Y_initial = self.model(torch.hstack((t_initial, x_initial)))
        Y_boundary = self.model(torch.hstack((t_boundary, x_boundary)))
        
        n_interior = Y_interior[:, 0].reshape(-1, 1)
        u_interior = Y_interior[:, 1].reshape(-1, 1)
        ϵ_interior = Y_interior[:, 2].reshape(-1, 1)
        
        n_initial = Y_initial[:, 0].reshape(-1, 1)
        u_initial = Y_initial[:, 1].reshape(-1, 1)
        ϵ_initial = Y_initial[:, 2].reshape(-1, 1)
        
        n_boundary = Y_boundary[:, 0].reshape(-1, 1)
        u_boundary = Y_boundary[:, 1].reshape(-1, 1)
        ϵ_boundary = Y_boundary[:, 2].reshape(-1, 1)
        
        n_x_interior = grad(n_interior, x_interior, grad_outputs=torch.ones_like(n_interior), retain_graph=True, create_graph=True)[0]
        n_t_interior = grad(n_interior, t_interior, grad_outputs=torch.ones_like(n_interior), retain_graph=True, create_graph=True)[0]
        
        n_xx_interior = grad(n_x_interior, x_interior, grad_outputs=torch.ones_like(n_x_interior), retain_graph=True, create_graph=True)[0]
        
        u_x_interior = grad(u_interior, x_interior, grad_outputs=torch.ones_like(u_interior), retain_graph=True, create_graph=True)[0]
        u_t_interior = grad(u_interior, t_interior, grad_outputs=torch.ones_like(u_interior), retain_graph=True, create_graph=True)[0]
        
        u_xx_interior = grad(u_x_interior, x_interior, grad_outputs=torch.ones_like(u_x_interior), retain_graph=True, create_graph=True)[0]
        
        ϵ_x_interior = grad(ϵ_interior, x_interior, grad_outputs=torch.ones_like(ϵ_interior), retain_graph=True, create_graph=True)[0]
        ϵ_t_interior = grad(ϵ_interior, t_interior, grad_outputs=torch.ones_like(ϵ_interior), retain_graph=True, create_graph=True)[0]
        
        ϵ_x_boundary = grad(ϵ_boundary, x_boundary, grad_outputs=torch.ones_like(ϵ_boundary), retain_graph=True, create_graph=True)[0]
        
        l_interior = l_0/(1 + l_0**2 * (n_x_interior - u_x_interior)**2 / ϵ_interior)**(κ/2)
        w_interior = C_χ * l_interior**2 * ϵ_interior / torch.sqrt(α**2 + a_u * u_interior**2)
        
        residual_n_t_loss = self.residual_n_t_loss(x_interior, n_t_interior, n_x_interior, n_xx_interior, ϵ_interior, l_interior)
        residual_u_t_loss = self.residual_u_t_loss(x_interior, ϵ_interior, n_x_interior, u_t_interior, u_xx_interior, l_interior, w_interior)
        residual_ϵ_t_loss = self.residual_ϵ_t_loss(x_interior, ϵ_interior, n_x_interior, u_x_interior, ϵ_t_interior, ϵ_x_interior, l_interior, w_interior)
        
        initial_n_loss = self.n_initial_loss(t_initial, x_initial, n_initial)
        initial_u_loss = self.u_initial_loss(t_initial, x_initial, u_initial)
        initial_ϵ_loss = self.ϵ_initial_loss(t_initial, x_initial, ϵ_initial)
        
        boundary_n_loss = self.n_boundary_loss(t_boundary, x_boundary, n_boundary)
        boundary_u_loss = self.u_boundary_loss(t_boundary, x_boundary, u_boundary)
        boundary_ϵ_x_loss = self.ϵ_x_boundary_loss(t_boundary, x_boundary, ϵ_x_boundary)
    
        total_loss = residual_n_t_loss + residual_u_t_loss + residual_ϵ_t_loss + initial_n_loss + initial_u_loss + initial_ϵ_loss + boundary_n_loss + boundary_u_loss + boundary_ϵ_x_loss
        
        return total_loss, residual_n_t_loss, residual_u_t_loss, residual_ϵ_t_loss, initial_n_loss, initial_u_loss, initial_ϵ_loss, boundary_n_loss, boundary_u_loss, boundary_ϵ_x_loss
    
    def train(self):
        if self.args.sine_initialisation:
            self.initialise_starting_surface()
        
        cwd = os.getcwd()
        plot_dirs = os.path.join(cwd, f'plots/{self.args.experiment_name}')

        if not os.path.isdir(plot_dirs):
            os.makedirs(plot_dirs)
        
        mlflow.set_experiment(self.args.experiment_name)
        mlflow.start_run()
        
        mlflow.log_param("physical_params", physical_params)
        mlflow.log_params(vars(self.args))
        mlflow.log_param("num_parameters", self.num_parameters)
        
        for epoch in tqdm(range(self.args.epochs), position=0, leave=True, desc='Training...', disable=True): 
            self.epoch = epoch
            
            # eval
            if epoch % self.args.eval_interval == 0 or epoch == self.args.epochs - 1:
                total_loss, residual_n_t_loss, residual_u_t_loss, residual_ϵ_t_loss, initial_n_loss, initial_u_loss, initial_ϵ_loss, boundary_n_loss, boundary_u_loss, boundary_ϵ_x_loss \
                    = self.forward(self.eval_X_interior, self.eval_X_initial, self.eval_X_boundary)
                
                print()
                print(f'Epoch: {self.epoch}, Loss: {total_loss.item():,.4e}')
                print(f"residual_n_t_loss: {residual_n_t_loss.item():.4e}, residual_u_t_loss: {residual_u_t_loss.item():.4e}, residual_ϵ_t_loss: {residual_ϵ_t_loss.item():.4e}")
                print(f"initial_n_loss: {initial_n_loss.item():.4e}, initial_u_loss: {initial_u_loss.item():.4e}, initial_ϵ_loss: {initial_ϵ_loss.item():.4e}")
                print(f"boundary_n_loss: {boundary_n_loss.item():.4e}, boundary_u_loss: {boundary_u_loss.item():.4e}, boundary_ϵ_x_loss: {boundary_ϵ_x_loss.item():.4e}")
                
                mlflow.pytorch.log_model(self.model, f"{self.args.experiment_name}_model_epoch_{self.epoch}")
                
                if self.args.plot_outputs:
                    self.plot_outputs()
        
            # training step
            self.optimizer.zero_grad()
            if epoch % self.args.sample_interval == 0:
                if self.args.multistage_training:
                    X_interior = self.sample_interior_points(epoch/self.args.epochs)
                else:
                    X_interior = self.sample_interior_points()
                    
                X_initial = self.sample_initial_points()
                X_boundary = self.sample_boundary_points()
            
            total_loss, residual_n_t_loss, residual_u_t_loss, residual_ϵ_t_loss, initial_n_loss, initial_u_loss, initial_ϵ_loss, boundary_n_loss, boundary_u_loss, boundary_ϵ_x_loss = self.forward(X_interior, X_initial, X_boundary)
            total_loss.backward()
            self.optimizer.step()
            
            mlflow.log_metric("total_loss", total_loss.item(), step=self.epoch)
            mlflow.log_metric("residual_n_t_loss", residual_n_t_loss.item(), step=self.epoch)
            mlflow.log_metric("residual_u_t_loss", residual_u_t_loss.item(), step=self.epoch)
            mlflow.log_metric("residual_ϵ_t_loss", residual_ϵ_t_loss.item(), step=self.epoch)
            mlflow.log_metric("initial_n_loss", initial_n_loss.item(), step=self.epoch)
            mlflow.log_metric("initial_u_loss", initial_u_loss.item(), step=self.epoch)
            mlflow.log_metric("initial_ϵ_loss", initial_ϵ_loss.item(), step=self.epoch)
            mlflow.log_metric("boundary_n_loss", boundary_n_loss.item(), step=self.epoch)
            mlflow.log_metric("boundary_u_loss", boundary_u_loss.item(), step=self.epoch)
            mlflow.log_metric("boundary_ϵ_x_loss", boundary_ϵ_x_loss.item(), step=self.epoch)
            
            if total_loss.isnan():
                print(f'Epoch: {self.epoch}, Loss: {total_loss.item():,.4e}')
                print(f"residual_n_t_loss: {residual_n_t_loss.item():.4e}, residual_u_t_loss: {residual_u_t_loss.item():.4e}, residual_ϵ_t_loss: {residual_ϵ_t_loss.item():.4e}")
                print(f"initial_n_loss: {initial_n_loss.item():.4e}, initial_u_loss: {initial_u_loss.item():.4e}, initial_ϵ_loss: {initial_ϵ_loss.item():.4e}")
                print(f"boundary_n_loss: {boundary_n_loss.item():.4e}, boundary_u_loss: {boundary_u_loss.item():.4e}, boundary_ϵ_x_loss: {boundary_ϵ_x_loss.item():.4e}")
                print("total_loss is NaN, stopping training...")
                mlflow.log_text("error", "total_loss is NaN, stopping training...")
                break
    
        mlflow.pytorch.log_model(self.model, f"{self.args.experiment_name}_model_final")
            
        if self.args.plot_outputs:
            self.save_gif()
        
        mlflow.end_run()
            
pinn = PINN(args)
pinn.train()