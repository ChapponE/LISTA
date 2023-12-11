import torch

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define A as standard Gaussian :
m = 250
n = 500
A = torch.normal(torch.zeros([m,n]), torch.tensor([1.0/m]))
  # Normalize colums to have unit l2 norm:
for i in range(A[0,:].shape[0]):
  norm_column = torch.norm(A[:, i], 2)
  A[:,i] /= norm_column

# Samples definition:
def gen_samples(nb_samples, noise=False, SNR=30):
  X_target = torch.normal(torch.zeros([nb_samples, n]), torch.tensor([1])).to(device)
    # Samples sparse :
  eps = torch.zeros([nb_samples, m], device = device)
  pb = 0.1
  coin = torch.bernoulli(pb*torch.ones([nb_samples, n])).to(device)
  X_target = X_target * coin

  if noise:
    eps = torch.normal(torch.zeros([nb_samples, m]), torch.pow(torch.tensor(10.0), torch.tensor(-SNR/20.0))).to(device)
  # Observed value:
  X_observed = torch.matmul(X_target, A.transpose(0,1)) + eps
  return(torch.utils.data.TensorDataset(X_observed, X_target.requires_grad_(True), torch.arange(nb_samples)))

'%%%%% Without noise %%%%'
#Train data et test data:
nb_samples_train = 1000
train_data = gen_samples(nb_samples_train)
test_data = gen_samples(nb_samples_train)

'%%%%% With noise %%%%'
#Train data et test data:
nb_samples_train = 1000
train_data_noisy = gen_samples(nb_samples_train, noise = True)
test_data_noisy = gen_samples(nb_samples_train, noise = True)

save = True
path = './'

if save:
  # Save in dictionary:
  data = {'sensor': A, 'train_data': train_data, 'test_data': test_data, 'train_data_noisy': train_data_noisy, 'test_data_noisy': test_data_noisy}
  torch.save(data, path+'data.pth')

# import data
data = torch.load(path+'data.pth')
A = data['sensor']
train_data = data['train_data']
test_data = data['test_data']
train_data_noisy = data['train_data_noisy']
test_data_noisy = data['test_data_noisy']

