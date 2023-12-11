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

model_names = ['lista_cp']
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

# FB and FB_adaptative:
from src.models.forward_backward import Forward_Backward
from src.models.forward_backward_adaptative import Forward_Backward_adaptative

partials_NMSE_fb = []
for lambd in [0.1, 0.05, 0.025]:
    fb_model = Forward_Backward(size_observed=size_observed, size_target=size_target, A=A, lambd=lambd, Niter=600, device=device)
    partials_NMSE = fb_model.__partial_NMSE__(train_data[:][0], train_data[:][1])
    partials_NMSE_fb.append([partials_NMSE[i].clone().detach().cpu().numpy() for i in range(len(partials_NMSE))])

fb_adaptative_model = Forward_Backward_adaptative(size_observed=size_observed, size_target=size_target, A=A, lambd=0.2, Niter=600, device=device)
partials_NMSE = fb_adaptative_model.__partial_NMSE__(train_data[:][0], train_data[:][1])
partials_NMSE_1_fb_adaptative = [partials_NMSE[i].clone().detach().cpu().numpy() for i in range(len(partials_NMSE))]

# Plot
#Plot des NMSE

fig = plt.figure(1, figsize=(9,4))
for i, model_to_train in enumerate(model_names):
    plt.plot(np.arange(lista_models[i].depth+1), partials_NMSE_Lista[i], label = model_to_train)
for i, lambd in enumerate([0.1, 0.05, 0.025]):
    plt.plot(np.arange(fb_model.Niter+1), partials_NMSE_fb[i], label = 'FB, $\lambda$='+str(lambd), linestyle = 'dashed' )
plt.plot(np.arange(fb_adaptative_model.Niter+1), partials_NMSE_1_fb_adaptative, label = 'FB_adaptative', linestyle = 'dashed' )

plt.title('NMSE/layers')
plt.xlabel('Iterations/Layers (k)')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.grid()
plt.show()
