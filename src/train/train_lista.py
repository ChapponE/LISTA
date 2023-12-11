
import torch
from torch.nn import MSELoss
from torch.optim import Adam, Adadelta

from torch.utils.data import DataLoader
from src.utils.trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

epochs = 2500
noise = True

# load the model
model_names = ['lista']
lista_models = []
lista_models_noisy = []
lista_trainer_histories = []
lista_trainer_noisy_histories = []
partials_NMSE_Lista = []
for i, model_to_train in enumerate(model_names):
    print('################', model_to_train, '################')
    if model_to_train == 'lista':
        from src.models.lista import LISTA_model as model
    elif model_to_train == 'lista_cp':
        from src.models.lista_cp import LISTACP_model as model
    elif model_to_train == 'lista_ss':
        from src.models.lista_ss import LISTASS_model as model
    elif model_to_train == 'lista_cpss':
        from src.models.lista_cpss import LISTACPSS_model as model

    # Create the neural network and move it to the computational device
    Lista_model = model( size_observed=size_observed, size_target=size_target, A=A, depth=16, init=True, device=device).to(device)
    Lista_model_noisy = model( size_observed=size_observed, size_target=size_target, A=A, depth=16, init=True, device=device).to(device)
    print("Number of parameters:", sum(w.numel() for w in Lista_model.parameters()))

    # Create the DataLoader and configure mini-batching
    batch_size = nb_samples
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    if noise:
        train_dataloader_noisy = DataLoader(train_data_noisy, batch_size=batch_size, shuffle=True)
        val_dataloader_noisy = DataLoader(test_data_noisy, batch_size=batch_size, shuffle=True)

    # Initialize loss, optimizer and trainer
    loss = MSELoss()
    a=0
    for i in Lista_model.parameters():
        a+=1
    print(a, list(Lista_model.parameters())[2] )
    # optimizer = Adam(Lista_model.parameters(), lr=1e-3)
    optimizer = Adadelta(Lista_model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, foreach=None, maximize=False)

    if noise:
        # optimizer_noisy = Adam(Lista_model_noisy.parameters(), lr=1e-3)
        optimizer_noisy = Adadelta(Lista_model_noisy.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, foreach=None, maximize=False)

    scheduler = False
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=3)

    Lista_trainer = Trainer(Lista_model, optimizer, scheduler, loss,
                      train_dataloader, val_dataloader, check_val_every_n_epoch=10)
    if noise:
        Lista_trainer_noisy = Trainer(Lista_model_noisy, optimizer_noisy, scheduler, loss,
                      train_dataloader_noisy, val_dataloader_noisy, check_val_every_n_epoch=10)

    # # Run training
    print(Lista_model.threshold)
    Lista_trainer.run(epochs)
    print(Lista_model.threshold)

    if noise:
        print(Lista_model_noisy.threshold)
        Lista_trainer_noisy.run(epochs)
        print(Lista_model_noisy.threshold)


    # sauvegarder les modèles
    torch.save(Lista_model.state_dict(), f'../../data/results/{model_to_train}_model.pth')
    if noise:
        torch.save(Lista_model_noisy.state_dict(), f'../../data/results/{model_to_train}_model_noisy.pth')

    # sauvegarder l'historique d'entraînement
    torch.save(Lista_trainer.history, f'../../data/results/{model_to_train}_trainer_history.pth')
    if noise:
        torch.save(Lista_trainer_noisy.history, f'../../data/results/{model_to_train}_trainer_noisy_history.pth')

