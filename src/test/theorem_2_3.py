import numpy as np
import torch
from matplotlib import pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
data = torch.load('../../data/data.pth')
A = data['sensor']
train_data = data['train_data']
test_data = data['test_data']
train_data_noisy = data['train_data_noisy']
test_data_noisy = data['test_data_noisy']

#nb_samples = train_data[:][0].shape[0]
nb_samples = len(train_data)
size_target = train_data[:][1].shape[1]
size_observed = train_data[:][0].shape[1]

model_names = ['lista', 'lista_cp', 'lista_ss', 'lista_cpss']
lista_models = []
lista_models_noisy = []
lista_trainer_histories = []
lista_trainer_noisy_histories = []
partials_NMSE_Lista = []
for i, model_to_train in enumerate(model_names):
    if model_to_train == 'lista':
        from src.models.lista import LISTA_model as model
    elif model_to_train == 'lista_cp':
        from src.models.lista_cp import LISTACP_model as model
    elif model_to_train == 'lista_ss':
        from src.models.lista_ss import LISTASS_model as model
    elif model_to_train == 'lista_cpss':
        from src.models.lista_cpss import LISTACPSS_model as model
    # charger les modèles
    lista_models.append(model(size_observed=size_observed, size_target=size_target, A=A, depth=16, init=True).to(device))
    (lista_models[i]).load_state_dict(torch.load(f'../../data/results/{model_to_train}_model.pth'))
    lista_models_noisy.append(model(size_observed=size_observed, size_target=size_target, A=A, depth=16, init=True).to(device))
    (lista_models_noisy[i]).load_state_dict(torch.load(f'../../data/results/{model_to_train}_model_noisy.pth'))
    lista_trainer_histories.append(torch.load(f'../../data/results/{model_to_train}_trainer_history.pth'))
    lista_trainer_noisy_histories.append(torch.load(f'../../data/results/{model_to_train}_trainer_noisy_history.pth'))

    # NMSE
    partials_NMSE = lista_models[i].__partial_NMSE__(train_data[:][0], train_data[:][1])
    partials_NMSE_Lista.append([partials_NMSE[i].clone().detach().cpu().numpy() for i in range(len(partials_NMSE))])

# FB and FFB:
from src.models.forward_backward import Forward_Backward
from src.models.fast_forward_backward import Fast_Forward_Backward

fb_model = Forward_Backward(size_observed=size_observed, size_target=size_target, A=A, lambd=0.1, Niter=16, device=device)
ffb_model = Fast_Forward_Backward(size_observed=size_observed, size_target=size_target, A=A, lambd=0.1, Niter=16, device=device)

# NMSE
partials_NMSE = fb_model.__partial_NMSE__(train_data[:][0], train_data[:][1])
partials_NMSE_fb = [partials_NMSE[i].clone().detach().cpu().numpy() for i in range(len(partials_NMSE))]
partials_NMSE = ffb_model.__partial_NMSE__(train_data[:][0], train_data[:][1])
partials_NMSE_ffb = [partials_NMSE[i].clone().detach().cpu().numpy() for i in range(len(partials_NMSE))]

# Plot
#Plot des NMSE

fig = plt.figure(1, figsize=(9,4))
for i, model_to_train in enumerate(model_names):
    plt.plot(np.arange(lista_models[i].depth+1), partials_NMSE_Lista[i], label = model_to_train)
plt.plot(np.arange(fb_model.Niter+1), partials_NMSE_fb, label = 'ISTA', linestyle = 'dotted' )
plt.plot(np.arange(fb_model.Niter+1), partials_NMSE_ffb, label = 'FISTA', linestyle = 'dotted' )

plt.title('NMSE/layers (SNR=inf)')
plt.xlabel('Iterations/Layers (k)')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.grid()
plt.show()