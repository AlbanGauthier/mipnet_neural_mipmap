import torch

from . import file_utils
from . import renderer

# from nvdiffmodelling
def dot(x: torch.Tensor, y: torch.Tensor, dim_) -> torch.Tensor:
    return torch.sum(x * y, dim=dim_, keepdim=True)

# from nvdiffmodelling
def length(x: torch.Tensor, dim=-1, eps: float = 1e-20) -> torch.Tensor:
	# Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
    return torch.sqrt(torch.clamp(dot(x, x, dim), min=eps))

# from nvdiffmodelling
def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def compute_normal_from_slopes(normals_xy):
	assert normals_xy.shape[-1] == 2
	norm = torch.clamp(length(normals_xy), min = 0.999)
	normals_xy = 0.999 * normals_xy / norm
	squared_xy = torch.square(normals_xy)
	if normals_xy.ndim == 2:
		z_vec = torch.sqrt(torch.clamp(
			1 - squared_xy[:, 0] - squared_xy[:, 1], 
			min=0.001))
		normal = torch.cat(
			(normals_xy, torch.unsqueeze(z_vec, dim=1)), 
			dim=1)
	else:
		z_vec = torch.sqrt(torch.clamp(
			1 - squared_xy[..., 0:1] - squared_xy[..., 1:2], 
			min=0.001))
		normal = torch.cat((normals_xy, z_vec), dim=-1)
	return normal


def reshape_maps_input_render(map_list):
	for i, map in enumerate(map_list):
		if map is not None:
			map_list[i] = torch.reshape(map, 
				(map.shape[0]*map.shape[1], map.shape[2]))
	return map_list


def batch_render(maps, wi_vec, wo_vec, light_col, 
	args, multi_wo, intensity=None):
	
	normal 		= maps[0]
	anisoMat 	= maps[1]
	height 		= maps[2]
	albedo 		= maps[3]
	metallic 	= maps[4]
	position2D 	= maps[5]
	
	if multi_wo:
		w_o = wo_vec
	else:
		w_o = renderer.Renderer.z_vector

	if albedo.ndim == 2:
		# expects T.ndim == 2 (H*W, N)
		render = renderer.Renderer.torch_Render(
			albedo, metallic, normal, anisoMat, 
			height, position2D, wi_vec, w_o, light_col, intensity)
	elif albedo.ndim == 3:
		dim = albedo.shape[0]
		maps = reshape_maps_input_render([
			albedo, metallic, normal, anisoMat, height, position2D])
		# expects T.ndim == 2 (H*W, N)
		render = renderer.Renderer.torch_Render(*maps, wi_vec, w_o, light_col, intensity)
		render = torch.reshape(render, 
			(dim,dim,3, wi_vec.shape[0]*w_o.shape[0])).squeeze()
	else:
		file_utils.exit_with_message("wrong map sizes in batch_render")
		
	assert not torch.any(torch.isfinite(render) == False)
	assert not torch.any(torch.isnan(render))
	
	return render


## Building an Orthonormal Basis, Revisited
def branchlessONB(n):
	sign = torch.sign(n[:,2])
	a = -1.0 / (sign + n[:,2])
	b = n[:,0] * n[:,1] * a
	b1 = torch.cat([
		1.0 + sign * n[:,0] * n[:,0] * a, 
		sign * b, -sign * n[:,0]], dim=1)
	b2 = torch.cat([
		b, sign + n[:,1] * n[:,1] * a, 
		-n[:,1]], dim=1)
	return b1, b2


def reinhardTonemapper(t):
	return t / (1 + t)


def neuMIPTonemapper(t):
	return torch.log(t + 1)


def gammaCorrection(input):
	"""linrgb2srgb"""
	limit = 0.0031308
	return torch.where(
		input > limit,
		1.055 * torch.pow(
			torch.clamp(
				input, min=limit), 
				(1.0 / 2.4)) - 0.055,
		12.92 * input)


def DeschaintrelogTensor(in_tensor):
	log_001 = torch.log(0.01)
	div_log = torch.log(1.01)-log_001
	return torch.log(in_tensor.add(0.01)).add(-log_001).div(div_log)


def process_raw_render(render):
	render = reinhardTonemapper(render)
	render = gammaCorrection(render)
	return render