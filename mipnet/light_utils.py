import numpy as np
import torch

from . import sphere_fibonacci_grid as fibo
from . import data_utils


def most_basic_generation(sample_count):
    # sample a set of random direction used for evaluating the loss
	omega_i = data_utils.torch_norm(torch.rand(sample_count, 3) * 2 - 1)

	# they all have their Z pointing up (otherwise the bsdf returns 0 anyways)
	omega_i[:,2].abs_()

	return torch.reshape(omega_i, (sample_count, 3))


def generate_fibonacci(ng):
    n_samples = 2 * ng # lower hemisphere is discarded
    xg = torch.from_numpy(
        fibo.sphere_fibonacci_grid_points(n_samples))
    selected = relocate_near_z(xg[xg[:,2] > 0.0])
    return selected


def generate_uniform_sphere(nb_samples, cosine_weighted = True):
    array = np.zeros((nb_samples, 3))
    for i in range(nb_samples):
        rand_s = np.random.random(2)
        theta = 2 * np.pi * rand_s[0]
        if cosine_weighted:
            phi = np.arccos(np.sqrt(1.0 - rand_s[1]))
        else:
            phi = np.arccos(1.0 - rand_s[1])
        array[i, 0] = np.cos(theta)*np.sin(phi)
        array[i, 1] = np.sin(theta)*np.sin(phi)
        array[i, 2] = np.cos(phi)
    return torch.from_numpy(array)


def relocate_near_z(wi_array, power = 0.5):
    wi_array[:, 2] = torch.where(
        wi_array[:, 2] < 0.1, 
        torch.pow(wi_array[:, 2], power), 
        wi_array[:, 2])
    wi_array = wi_array / torch.linalg.norm(
        wi_array, keepdims=True, axis = 1)
    return wi_array


def radicalInverse_VdC(bits):
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return float(bits) * 2.3283064365386963e-10


def hammersley2d(i, N):
    return np.array([float(i)/float(N), radicalInverse_VdC(i)])

def generate_hammersley(n, cosine_weighted):
    plane_coords = np.array([hammersley2d(i, n) for i in range(n)])
    if cosine_weighted:
        theta = np.arccos(np.sqrt(1 - plane_coords[:, 0]))
    else:
        theta = np.arccos(1 - plane_coords[:, 0])
    phi = 2 * np.pi * plane_coords[:, 1]
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return torch.from_numpy(np.stack([x,y,z], axis=1))


def generate_test_set_hammersley(n, cosine_weighted):
    plane_coords = np.array([hammersley2d(i, n) for i in range(n)])
    plane_coords = plane_coords + np.array([0.125, 0.125])
    plane_coords = np.modf(plane_coords)[0]
    if cosine_weighted:
        theta = np.arccos(np.sqrt(1 - plane_coords[:, 0]))
    else:
        theta = np.arccos(1 - plane_coords[:, 0])
    phi = 2 * np.pi * plane_coords[:, 1]
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return torch.from_numpy(np.stack([x,y,z], axis=1))


def generate_hemisphere_pts(sampling, sample_count, near_z, cosine_weighted = False):
    if sampling == "Hammersley":
        samples = generate_hammersley(sample_count, cosine_weighted)
        if near_z:
            samples = relocate_near_z(samples)
        return samples
    elif sampling == "Fibonacci":
        samples = generate_fibonacci(sample_count)
        if near_z:
            samples = relocate_near_z(samples)
        return samples
    else:
        print("wrong sampling type")
        exit(0)
