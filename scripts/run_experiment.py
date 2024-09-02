import subprocess

args = {
    '--experiment_name': 'test',
    '--layers': 4,
    '--width': 64,
    '--epochs': 100000,
    '--eval_interval': 2500,
    '--interior_batch_size': 2048,
    '--initial_batch_size': 2048,
    '--boundary_batch_size': 2048,
    '--x_min': 0.0,
    '--x_max': 1.0,
    '--t_min': 0.0,
    '--t_max': 1.0,
    '--sample_interval': 1000,
    '--interior_loss_function': 'l2',
    '--initial_loss_function': 'linf',
    '--boundary_loss_function': 'linf',
    '--force_positive_method': 'exp',
    '--optimizer': 'adam',
    '--device': 'cuda',
    '--precision': 'float64',
    '--sine_initialisation': True,
    '--sine_initialisation_epochs': 100000,
    '--sine_initialisation_learning_rate': 1e-3,
    '--sine_initialisation_eval_interval': 5000,
    '--multistage_training': False,
    '--plot_outputs': True,
    '--plot_vars': 'n, n_t, n_x, n_xx, u, u_t, u_x, u_xx, ϵ, ϵ_t, ϵ_x'
}

arglist = []

for k, v in args.items():
    arglist.append(k)
    arglist.append(str(v))

command = ['python', 'train_fusion_pinn.py'] + arglist

print(f"executing {command}")

f = open(f"logs/test.log", "w")
subprocess.call(command, stdout=f, stderr=f)