import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss

### define the model to train ###
model_to_train = 'lista'
##################################
if model_to_train == 'lista':
    from src.models.lista import LISTA_model as model
elif model_to_train == 'lista_cp':
    from src.models.lista_cp import LISTACP_model as model
elif model_to_train == 'lista_ss':
    from src.models.lista_ss import LISTASS_model as model
elif model_to_train == 'lista_cpss':
    from src.models.lista_cpss import LISTACPSS_model as model


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


# charger les modèles
Lista_model = model(size_observed=size_observed, size_target=size_target, A=A, depth=16, init=True).to(device)
Lista_model.load_state_dict(torch.load(f'../../data/results/{model_to_train}_model.pth'))
Lista_model_noisy = model(size_observed=size_observed, size_target=size_target, A=A, depth=16, init=True).to(device)
Lista_model_noisy.load_state_dict(torch.load(f'../../data/results/{model_to_train}_model_noisy.pth'))
Lista_trainer_history = torch.load(f'../../data/results/{model_to_train}_trainer_history.pth')
Lista_trainer_noisy_history = torch.load(f'../../data/results/{model_to_train}_trainer_noisy_history.pth')

loss = MSELoss()

# Learning curve whithout noise
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for i, mode in enumerate(Lista_trainer_history.keys()):
    data = dict()
    for epoch, record in Lista_trainer_history[mode]:
        data.setdefault("epoch", []).append(epoch)
        for metric, value in record.items():
            data.setdefault(metric, []).append(value)

    ax[0].plot(data["epoch"], data["Loss"], label=mode)
    ax[0].set_title(f"{mode} loss")
    ax[0].set_xlabel('number of training iteration')
    ax[0].set_ylabel('MSE')
    ax[0].grid()

    ax[1].plot(data["epoch"], data["PSNR"], label=mode)
    ax[1].set_title(f"{mode} PSNR")
    ax[1].set_xlabel('number of training iteration')
    ax[1].set_ylabel('PSNR')
    ax[1].grid()

fig.suptitle(f'{model_to_train} without noise', fontsize=25, fontweight='bold')
# ax[0].set_yscale('log')
# ax[1].set_yscale('log')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()

# learning curve with noise
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for i, mode in enumerate(Lista_trainer_noisy_history.keys()):
    data = dict()
    for epoch, record in Lista_trainer_noisy_history[mode]:
        data.setdefault("epoch", []).append(epoch)
        for metric, value in record.items():
            data.setdefault(metric, []).append(value)

    ax[0].plot(data["epoch"], data["Loss"], label=mode)
    ax[0].set_title(f"{mode} loss")
    ax[0].set_xlabel('number of training iteration')
    ax[0].set_ylabel('MSE')
    ax[0].grid()

    ax[1].plot(data["epoch"], data["PSNR"], label=mode)
    ax[1].set_title(f"{mode} PSNR")
    ax[1].set_xlabel('number of training iteration')
    ax[1].set_ylabel('PSNR')
    ax[1].grid()

fig.suptitle(f'{model_to_train} with noise', fontsize=25, fontweight='bold')
# ax[0].set_yscale('log')
# ax[1].set_yscale('log')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()

