import subprocess
import os
from itertools import product
from tqdm.auto import tqdm

interior_loss_function = ['l1', 'l2']
initial_boundary_loss_function = ['l1', 'linf']
layers = [4]
width = [64]
learning_rate = [1e-3, 1e-4]
sine_initialisation = [True, False]
multistage_training = [True, False]
separate_models = [True, False]
force_positive_method = ['exp', 'abs']

params = [interior_loss_function, initial_boundary_loss_function, layers, width, learning_rate, sine_initialisation, multistage_training, separate_models, force_positive_method]

print(f"Number of experiments: {len(list(product(*params)))}")

cwd = os.getcwd()
log_dir = os.path.join(cwd, "logs")

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

for param in tqdm(product(*params)):
    interior_loss_function = param[0]
    initial_boundary_loss_function = param[1]
    layers = param[2]
    width = param[3]
    learning_rate = param[4]
    sine_initialisation = param[5]
    multistage_training = param[6]
    separate_models = param[7]
    force_positive_method = param[8]
    
    experiment_name = f"intloss_{interior_loss_function}_boundloss_{initial_boundary_loss_function}_layers_{layers}_width_{width}_lr_{learning_rate}_sine_{sine_initialisation}_multistage_{multistage_training}_separate_{separate_models}_force_{force_positive_method}"

    args = {
        '--experiment_name': experiment_name,
        '--layers': layers,
        '--width': width,
        '--learning_rate': learning_rate,
        '--epochs': 100000,
        '--eval_interval': 2500,
        '--interior_batch_size': 2048,
        '--initial_batch_size': 2048,
        '--boundary_batch_size': 2048,
        '--x_min': 0.0,
        '--x_max': 1.0,
        '--t_min': 0.0,
        '--t_max': 1.0,
        '--sample_interval': 500,
        '--interior_loss_function': interior_loss_function,
        '--initial_loss_function': initial_boundary_loss_function,
        '--boundary_loss_function': initial_boundary_loss_function,
        '--force_positive_method': force_positive_method,
        '--optimizer': 'adam',
        '--device': 'cuda',
        '--precision': 'float64',
        '--sine_initialisation': sine_initialisation,
        '--sine_initialisation_epochs': 100000,
        '--sine_initialisation_learning_rate': 1e-3,
        '--sine_initialisation_eval_interval': 5000,
        '--multistage_training': multistage_training,
        '--separate_models': separate_models,
        '--plot_outputs': True,
        '--plot_vars': 'n, n_t, n_x, n_xx, u, u_t, u_x, u_xx, ϵ, ϵ_t, ϵ_x'
    }

    arglist = []

    for k, v in args.items():
        arglist.append(k)
        arglist.append(str(v))

    command = ['python', 'train_fusion_pinn.py'] + arglist

    print()
    print("-------------------------")
    print(f"executing {command}")
    print("-------------------------")
    
    f = open(f"logs/{experiment_name}.log", "w")
    
    process = subprocess.Popen(command, stdout=f, stderr=f)
    out = process.communicate()