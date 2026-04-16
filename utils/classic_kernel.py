# '''Implementation of kernel functions.'''

# import torch


# def euclidean_distances(samples, centers, squared=True):
#     samples_norm = torch.sum(samples**2, dim=1, keepdim=True)
#     if samples is centers:
#         centers_norm = samples_norm
#     else:
#         centers_norm = torch.sum(centers**2, dim=1, keepdim=True)
#     centers_norm = torch.reshape(centers_norm, (1, -1))

#     distances = samples.mm(torch.t(centers))
#     distances.mul_(-2)
#     distances.add_(samples_norm)
#     distances.add_(centers_norm)
#     #print(centers_norm.size(), samples_norm.size(), distances.size())
#     if not squared:
#         distances.clamp_(min=0)
#         distances.sqrt_()

#     return distances


# def euclidean_distances_M(samples, centers, M, squared=True):

#     samples_norm = (samples @ M)  * samples
#     samples_norm = torch.sum(samples_norm, dim=1, keepdim=True)

#     if samples is centers:
#         centers_norm = samples_norm
#     else:
#         centers_norm = (centers @ M) * centers
#         centers_norm = torch.sum(centers_norm, dim=1, keepdim=True)

#     centers_norm = torch.reshape(centers_norm, (1, -1))

#     distances = samples.mm(M @ torch.t(centers))
#     distances.mul_(-2)
#     distances.add_(samples_norm)
#     distances.add_(centers_norm)

#     if not squared:
#         distances.clamp_(min=0)
#         distances.sqrt_()

#     return distances


# def gaussian(samples, centers, bandwidth):
#     '''Gaussian kernel.

#     Args:
#         samples: of shape (n_sample, n_feature).
#         centers: of shape (n_center, n_feature).
#         bandwidth: kernel bandwidth.

#     Returns:
#         kernel matrix of shape (n_sample, n_center).
#     '''
#     assert bandwidth > 0
#     kernel_mat = euclidean_distances(samples, centers)
#     kernel_mat.clamp_(min=0)
#     gamma = 1. / (2 * bandwidth ** 2)
#     kernel_mat.mul_(-gamma)
#     kernel_mat.exp_()

#     #print(samples.size(), centers.size(),
#     #      kernel_mat.size())
#     return kernel_mat


# def laplacian(samples, centers, bandwidth):
#     '''Laplacian kernel.

#     Args:
#         samples: of shape (n_sample, n_feature).
#         centers: of shape (n_center, n_feature).
#         bandwidth: kernel bandwidth.

#     Returns:
#         kernel matrix of shape (n_sample, n_center).
#     '''
#     assert bandwidth > 0
#     kernel_mat = euclidean_distances(samples, centers, squared=False)
#     kernel_mat.clamp_(min=0)
#     gamma = 1. / bandwidth
#     kernel_mat.mul_(-gamma)
#     kernel_mat.exp_()
#     return kernel_mat



# def laplacian_M(samples, centers, bandwidth, M):
#     assert bandwidth > 0
#     kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
#     kernel_mat.clamp_(min=0)
#     gamma = 1. / bandwidth
#     kernel_mat.mul_(-gamma)
#     kernel_mat.exp_()
#     return kernel_mat


# def dispersal(samples, centers, bandwidth, gamma):
#     '''Dispersal kernel.

#     Args:
#         samples: of shape (n_sample, n_feature).
#         centers: of shape (n_center, n_feature).
#         bandwidth: kernel bandwidth.
#         gamma: dispersal factor.

#     Returns:
#         kernel matrix of shape (n_sample, n_center).
#     '''
#     assert bandwidth > 0
#     kernel_mat = euclidean_distances(samples, centers)
#     kernel_mat.pow_(gamma / 2.)
#     kernel_mat.mul_(-1. / bandwidth)
#     kernel_mat.exp_()
#     return kernel_mat

'''Implementation of kernel functions. (Refactored for Autograd support)'''

import torch

def euclidean_distances(samples, centers, squared=True):
    samples_norm = torch.sum(samples**2, dim=1, keepdim=True)
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = torch.sum(centers**2, dim=1, keepdim=True)
    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(torch.t(centers))
    
    # Replaced in-place .mul_() and .add_()
    distances = distances * -2
    distances = distances + samples_norm
    distances = distances + centers_norm
    
    if not squared:
        # Replaced in-place .clamp_() and .sqrt_()
        distances = torch.clamp(distances, min=0)
        distances = torch.sqrt(distances)

    return distances


def euclidean_distances_M(samples, centers, M, squared=True):

    samples_norm = (samples @ M)  * samples
    samples_norm = torch.sum(samples_norm, dim=1, keepdim=True)

    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = (centers @ M) * centers
        centers_norm = torch.sum(centers_norm, dim=1, keepdim=True)

    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(M @ torch.t(centers))
    
    # Replaced in-place .mul_() and .add_()
    distances = distances * -2
    distances = distances + samples_norm
    distances = distances + centers_norm

    if not squared:
        # Replaced in-place .clamp_() and .sqrt_()
        distances = torch.clamp(distances, min=0)
        distances = torch.sqrt(distances)

    return distances


def gaussian(samples, centers, bandwidth):
    '''Gaussian kernel.'''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)
    
    # Replaced in-place operations
    kernel_mat = torch.clamp(kernel_mat, min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat = kernel_mat * -gamma
    kernel_mat = torch.exp(kernel_mat)

    return kernel_mat


def laplacian(samples, centers, bandwidth):
    '''Laplacian kernel.'''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    
    # Replaced in-place operations
    kernel_mat = torch.clamp(kernel_mat, min=0)
    gamma = 1. / bandwidth
    kernel_mat = kernel_mat * -gamma
    kernel_mat = torch.exp(kernel_mat)
    return kernel_mat



def laplacian_M(samples, centers, bandwidth, M):
    assert bandwidth > 0
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    
    # Replaced in-place operations
    kernel_mat = torch.clamp(kernel_mat, min=0)
    gamma = 1. / bandwidth
    kernel_mat = kernel_mat * -gamma
    kernel_mat = torch.exp(kernel_mat)
    return kernel_mat


def dispersal(samples, centers, bandwidth, gamma):
    '''Dispersal kernel.'''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)
    
    # Replaced in-place operations
    kernel_mat = torch.pow(kernel_mat, gamma / 2.)
    kernel_mat = kernel_mat * (-1. / bandwidth)
    kernel_mat = torch.exp(kernel_mat)
    return kernel_mat