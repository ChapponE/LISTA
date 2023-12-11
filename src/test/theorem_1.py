import numpy as np
import torch
from matplotlib import pyplot as plt


### define the model to train ###
model_to_train = 'lista'
##################################
if model_to_train == 'lista':
    from src.models.lista import LISTA_model as model
elif model_to_train == 'lista_ss':
    from src.models.lista_ss import LISTASS_model as model



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


with torch.no_grad():
    list_parameters = list(Lista_model.parameters())
    thetas = Lista_model.threshold
    list_weights = [ Lista_model.weight_layers[15-i].weight - torch.eye(Lista_model.size_target).to(device) + torch.matmul(Lista_model.bias_layers[15-i].weight, A) for i in range(Lista_model.depth)]
    list_weights_norm = [torch.linalg.norm(list_weights[i]).detach().item() for i in range(len(list_weights))]
print(list_weights[0].shape, Lista_model.bias_layers[0].weight.shape )
'%%%% noisy data %%%%'
with torch.no_grad():
    list_parameters_noisy = list(Lista_model_noisy.parameters())
    thetas_noisy = Lista_model_noisy.threshold
    list_weights_noisy = [ Lista_model_noisy.weight_layers[15-i].weight - torch.eye(Lista_model.size_target).to(device) + torch.matmul(Lista_model_noisy.bias_layers[15-i].weight, A) for i in range(Lista_model_noisy.depth)]
    list_weights_norm_noisy = [torch.linalg.norm(list_weights_noisy[i], dim=(0,1), ord=2).detach().item() for i in range(len(list_weights_noisy))]



fig = plt.figure(1, figsize=(12,6))
plt.subplot(222)
plt.plot(np.arange(Lista_model.depth), thetas.detach().cpu().numpy())
plt.title('theta')
plt.xlabel('Numbers of layers (k)')
plt.ylabel('theta (threshold)')

plt.subplot(221)
plt.plot(np.arange(Lista_model.depth), np.array(list_weights_norm))
plt.title('norm of weight')
plt.xlabel('number of layers (k)')
plt.ylabel('norm of weight-(I-bias*A)')



'%%%% noisy %%%%'
fig = plt.figure(1, figsize=(12,6))
plt.subplot(224)
plt.plot(np.arange(Lista_model_noisy.depth), thetas_noisy.detach().cpu().numpy())
plt.title('theta')
plt.xlabel('Numbers of layers (k)')
plt.ylabel('theta (threshold)')

plt.subplot(223)
plt.plot(np.arange(Lista_model_noisy.depth), np.array(list_weights_norm_noisy))
plt.title('norm of weight')
plt.xlabel('number of layers (k)')
plt.ylabel('norm of weight-(I-bias*A)')
plt.tight_layout()
plt.show()