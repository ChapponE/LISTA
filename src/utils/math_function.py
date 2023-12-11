import torch

#Gram matrix and max eigenvalue:
def max_eig(A):
  gram_A = torch.matmul(A.transpose(0,1),A)
  eigenvalues, eigenvectors = torch.linalg.eig(gram_A)
  L = torch.max(torch.abs(eigenvalues))*1.001
  return(gram_A, L)


"""**PSNR SNR:**"""

def PSNR(img1, img2):
    # Optional: get last 3 dimensions (CxNxM) so that to be compatible with single sample
    #dims = list(range(max(0, img1.ndim - 3), img1.ndim))
    # otherwise simply
    dims = list(range(1, img1.ndim))

    # Take care to the mean along only the last 3 dimensions.
    return 10 * torch.log10(1. / (img1 - img2).pow(2).mean(dims))

def SNR(img1, img2):
    dims = list(range(1, img1.ndim))
    return 20 * torch.log2(1. / (img1 - img2).pow(2).mean(dims))

