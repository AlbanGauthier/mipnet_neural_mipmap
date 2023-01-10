import torch
import os
import cv2
import numpy as np

from . import renderer
from . import render_utils
from . import file_utils
from . import model


def cart2sph(xyz):
	if xyz.ndim == 2:
		xy = xyz[:, 0]**2 + xyz[:, 1]**2
		theta = torch.atan2(torch.sqrt(xy), xyz[:, 2])
		phi = torch.atan2(xyz[:, 1], xyz[:, 0])
	else:
		xy = xyz[:, :, 0]**2 + xyz[:, :, 1]**2
		theta = torch.atan2(torch.sqrt(xy), xyz[:, :, 2])
		phi = torch.atan2(xyz[:, :, 1], xyz[:, :, 0])
	return theta, phi


def generate_rand_light_col():
	## output of size (1,1,3,1)
	
	A = 360 * torch.rand((1,1,1))
	B = torch.ones((1,1,2))
	tmp = torch.cat([A, B], dim=2)
	tmp = cv2.cvtColor(tmp.numpy(), cv2.COLOR_HSV2RGB)

	return torch.from_numpy(tmp).unsqueeze(-1)


def compute_2D_position_grid(size):
	range_vec = 2 * (np.arange(size) / size + 1 / (2 * size)) - 1
	X, Y = np.meshgrid(range_vec, range_vec)
	coords = torch.cat((
		torch.from_numpy(X).unsqueeze(-1),
		torch.from_numpy(Y).unsqueeze(-1)), dim=-1)
	return coords


def compute_3D_position_grid(size):
	range_vec = 2 * (np.arange(size) / size + 1 / (2 * size)) - 1
	X, Y = np.meshgrid(range_vec, range_vec)
	zeros = torch.zeros((size,size,1))
	coords = torch.cat((
		torch.from_numpy(X).unsqueeze(-1),
		torch.from_numpy(Y).unsqueeze(-1), zeros), dim=-1)
	return coords


def average_patches(in_vec, ks = 2, dim1 = 0, dim2 = 1):
	assert in_vec.shape[dim1] == in_vec.shape[dim2]
	in_vec = in_vec.unfold(dim1, ks, ks).unfold(dim2, ks, ks) 
	in_vec = torch.sum(in_vec, dim=-1)
	in_vec = torch.sum(in_vec, dim=-1)
	in_vec = in_vec / (ks * ks)
	return in_vec


def average_maps(map_list, ks):
	for i, map in enumerate(map_list):
		map_list[i] = average_patches(map, ks)
	map_list[0] = torch_norm(map_list[0]) # normal
	return map_list


def torch_norm(tensor, dimToNorm=2, eps=1e-20):
	assert tensor.shape[dimToNorm] == 3
	length = torch.sqrt(torch.clamp(
		torch.sum(torch.square(tensor), 
			axis=dimToNorm, keepdim=True), min=eps))
	return torch.div(tensor, length)


def normal_from_data_vector(data_vec):
	
	normals = render_utils.compute_normal_from_slopes(data_vec[..., 0:2])

	if normals.ndim == 2:
		normals = torch_norm(normals, dimToNorm=1)
	else:
		normals = torch_norm(normals)
	
	return normals


def get_renorm_clamped_maps(in_maps):
	
	# renormalize normals
	if in_maps[0].ndim == 2:
		in_maps[0] = torch_norm(in_maps[0], dimToNorm=1)
	else:
		in_maps[0] = torch_norm(in_maps[0])

	anisoMat = in_maps[1]
	aniso_a = anisoMat[..., 0:1]
	aniso_b = anisoMat[..., 1:2]
	aniso_c = anisoMat[..., 2:]

	min_alpha = renderer.Renderer.min_alpha

	# clamp aniso a & b
	aniso_a = torch.clamp(aniso_a, min=min_alpha, max=1.0)
	aniso_b = torch.clamp(aniso_b, min=min_alpha, max=1.0)

	aniso_aXb = aniso_a * aniso_b
	m_2 = renderer.Renderer.min_alpha_sqr
	
	bound_c_sqr = torch.min(aniso_aXb - m_2, (1 - aniso_a)*(1 - aniso_b))
	bound_c_sqr = torch.min((aniso_a - min_alpha)*(aniso_b - min_alpha), bound_c_sqr)
	bound_c_sqr = torch.clamp(bound_c_sqr, min=1e-20)
	bound_c = torch.pow(bound_c_sqr, 0.5)

	# sqrt aniso c
	aniso_c = torch.clamp(aniso_c, min=-bound_c, max=bound_c)

	in_maps[1] = torch.cat([aniso_a, aniso_b, aniso_c], dim=-1)

	idx = 2

	#height
	idx += 1

	# albedo
	in_maps[idx] = torch.clamp(in_maps[idx], min=0, max=1)

	# metallic
	in_maps[idx+1] = torch.clamp(in_maps[idx+1], min=0, max=1)

	#position
	if idx+2 != len(in_maps) - 1:
		file_utils.exit_with_message("error in get_renorm_clamped_maps")

	return in_maps


def maps_from_data(data_vec):
	"""Return dataloader like map order"""
	ndim = 2
	ret_list = []
	
	#normal
	ret_list.append(normal_from_data_vector(data_vec))
	# sqrt(A) matrix (a,b,c)
	ret_list.append(data_vec[..., ndim:ndim+3])
	ndim += 2
	#height
	ret_list.append(torch.unsqueeze(data_vec[..., ndim+1], dim=-1))
	#albedo
	ret_list.append(data_vec[..., ndim+2:ndim+5])
	#metallic
	ret_list.append(torch.unsqueeze(data_vec[..., ndim+5], dim=-1))
	#position
	ret_list.append(data_vec[..., ndim+6:])
	return ret_list


def add_unused_maps_to_mlp_output(mlp_output):

	mlp_output = torch.cat(
		[mlp_output, 
		0.0 * mlp_output[..., :3], #albedo
		0.0 * mlp_output[..., :1], #metallic
		0.0 * mlp_output[..., :2], #position
		], dim=-1)

	return mlp_output


def remove_unused_channels(input_data):

	output = input_data.clone()
	output = output[..., :-6] # albedo/metallic/position

	return output


def swap_tiles_and_channels(img_data):

	if img_data.ndim == 4:
		img_data = torch.permute(img_data, (0, 3, 1, 2))
	elif img_data.ndim == 3:
		img_data = torch.permute(img_data, (2, 0, 1))
		img_data = torch.unsqueeze(img_data, dim=0)
	else:
		assert False

	return img_data


def add_empty_output_height(mlp_out):
	return torch.cat([mlp_out, 0.0 * mlp_out[:, :1, ...]], dim=1)


def clamp_maps_into_valid_ranges(img_data):

	if img_data.ndim == 4:
		normal_xy = img_data[:, :2, ...].clone()
		aniso_a = img_data[:, 2:3, ...].clone()
		aniso_b = img_data[:, 3:4, ...].clone()
		aniso_c = img_data[:, 4:5, ...].clone()
		data_dim = 1
	elif img_data.ndim == 3:
		normal_xy = img_data[..., :2].clone()
		aniso_a = img_data[..., 2:3].clone()
		aniso_b = img_data[..., 3:4].clone()
		aniso_c = img_data[..., 4:5].clone()
		data_dim = 2
	else:
		return None

	assert normal_xy.shape[data_dim] == 2
	
	norm = torch.clamp(render_utils.length(normal_xy, dim=data_dim), min=0.999)
	
	#normal_xy
	normal_xy = 0.999 * normal_xy / norm

	min_alpha = renderer.Renderer.min_alpha

	# clamp aniso a & b
	aniso_a = torch.clamp(aniso_a, min=min_alpha, max=1.0)
	aniso_b = torch.clamp(aniso_b, min=min_alpha, max=1.0)

	aniso_aXb = aniso_a * aniso_b
	m_2 = renderer.Renderer.min_alpha_sqr
	
	bound_c_sqr = torch.min(aniso_aXb - m_2, (1 - aniso_a)*(1 - aniso_b))
	bound_c_sqr = torch.min((aniso_a - min_alpha)*(aniso_b - min_alpha), bound_c_sqr)
	bound_c_sqr = torch.clamp(bound_c_sqr, min=1e-20)
	bound_c = torch.pow(bound_c_sqr, 0.5)

	# sqrt aniso c
	aniso_c = torch.clamp(aniso_c, min=-bound_c, max=bound_c)

	if img_data.ndim == 4:
		return torch.cat([normal_xy, aniso_a, aniso_b, aniso_c, 
			img_data[:, 5:, ...]], dim=data_dim)
	elif img_data.ndim == 3:
		return torch.cat([normal_xy, aniso_a, aniso_b, aniso_c, 
			img_data[..., 5:]], dim=data_dim)


def process_in_mlp_multi_loss(L0, model_A, model_B, nb_levels):

	mlp_Li_delta_list = []
	mlp_Li_bar_list = []

	L0 = remove_unused_channels(L0)
	L0 = swap_tiles_and_channels(L0)
	# L0: (N, dims_to_learn, k, k)

	L1_bar = average_patches(L0, ks=2, dim1=2, dim2=3)

	# 256x256 patch
	mlp_L1_delta = model_A(L0)

	assert not torch.any(torch.isfinite(mlp_L1_delta) == False)
	assert not torch.any(torch.isnan(mlp_L1_delta))

	mlp_L1_delta = add_empty_output_height(mlp_L1_delta)
	# mlp_L1: (N, dims_to_learn, k/2, k/2)

	mlp_Li_delta_list.append(mlp_L1_delta)
	L1 = clamp_maps_into_valid_ranges(mlp_L1_delta + L1_bar)

	Li 	= L0
	Li1 = L1

	for _ in range(nb_levels - 1):
		
		mlp_Li2_bar = average_patches(Li1, ks=2, dim1=2, dim2=3)
		mlp_Li_bar_list.append(mlp_Li2_bar)

		mlp_Li2_tilde_delta = model_A(Li1)
		mlp_Li2_tilde_delta = add_empty_output_height(mlp_Li2_tilde_delta)

		Li2_tilde = clamp_maps_into_valid_ranges(mlp_Li2_tilde_delta + mlp_Li2_bar)
		
		mlp_Li2_delta = model_B(Li, Li1, Li2_tilde)
		mlp_Li2_delta = add_empty_output_height(mlp_Li2_delta)

		mlp_Li_delta_list.append(mlp_Li2_delta)
		Li2 = clamp_maps_into_valid_ranges(mlp_Li2_delta + mlp_Li2_bar)

		Li 	= Li1
		Li1 = Li2

	for i in range(len(mlp_Li_delta_list)):
		mlp_Li_delta_list[i] = torch.permute(mlp_Li_delta_list[i], (0, 2, 3, 1))
	
	for i in range(len(mlp_Li_bar_list)):
		mlp_Li_bar_list[i] = torch.permute(mlp_Li_bar_list[i], (0, 2, 3, 1))

	return mlp_Li_delta_list, mlp_Li_bar_list


def process_full_img_data_in_mlp(model, L0, L1 = None, L2 = None):

	L0 = remove_unused_channels(L0)
	L0 = swap_tiles_and_channels(L0)
	if L1 is not None:
		L1 = remove_unused_channels(L1)
		L1 = swap_tiles_and_channels(L1)
	if L2 is not None:
		L2 = remove_unused_channels(L2)
		L2 = swap_tiles_and_channels(L2)

	# L0: (N, dims_to_learn, tile_width, tile_width)
	model_output = model(L0, L1, L2)
	model_output = add_empty_output_height(model_output)

	# model_output: (N, dims_to_learn, tile_width // ks, tile_width // ks)
	model_output = torch.permute(model_output, (0, 2, 3, 1))

	# if batch_dim(N) = 1 then squeeze
	model_output = torch.squeeze(model_output)

	# model_output: (tile_width // ks, tile_width // ks, dims_to_learn)
	return model_output


def fold_tensor(input):
	## input: (width * width, ks, ks, channels)
	assert input.shape[1] == input.shape[2]
	width = int(np.sqrt(input.shape[0]))
	channels = input.shape[-1]
	ks = input.shape[1]
	input = input.reshape(width, width, ks, ks, channels)
	input = input.permute(0, 2, 1, 3, 4)
	input = input.reshape(ks * width, ks * width, channels)
	## output: (ks * width, ks * width, channels)
	return input


def compute_L1_gt_render(
	L0, wi_vec, wo_vec, light_col, 
	args, light_intensity = None):
	maps = maps_from_data(L0)
	gt_render = render_utils.batch_render(
		maps, wi_vec, wo_vec, light_col, args,
		multi_wo=args.multi_wo, intensity=light_intensity)
	gt_render = average_patches(gt_render, ks=2)
	return gt_render


def save_references(args, dataset, level_str):
	out_path = args.output_dir
	mat_idx = args.train_out_id
	name = dataset.get_material_name(mat_idx)
	img_data = dataset.get_material_data(mat_idx)
	ref_render = compute_full_lod_gt_render(
		int(level_str), img_data, 
		renderer.Renderer.z_vector, 
		renderer.Renderer.z_vector,
		renderer.Renderer.white_light, 
		args)
	ref_render = render_utils.process_raw_render(
		ref_render).cpu().numpy()
	if not os.path.isdir(out_path + name):
		os.makedirs(out_path + "0" + name, exist_ok=True)
	file_utils.write_image_as_png(
		out_path + "0" + name + "_" 
		+ level_str + "_zrender_ref.png", 
		ref_render, args.verbose)
	return


def add_missing_channels_to_mlp_out(mlp_out, L_bar):
	
	# add albedo/metallic/position
	mlp_out = torch.cat((mlp_out, L_bar[..., -6:]), dim=-1)
	
	return mlp_out


def render_maps_outdataVec_from_mlp_out(
	L_bar, mlp_L, wi_vec, wo_array, light_col, 
	args, multi_wo, light_intensity = None):
	
	delta_L = add_unused_maps_to_mlp_output(mlp_L)
	outdataVec = clamp_maps_into_valid_ranges(L_bar + delta_L)

	maps = maps_from_data(outdataVec)
	# maps = get_renorm_clamped_maps(maps, args)

	render = render_utils.batch_render(maps, wi_vec, wo_array,
		light_col, args, multi_wo, light_intensity)
	
	return render, maps, outdataVec


def tonemapped_render_from_mlp_output(mlp_L, L_bar, wi_array, wo_array,
	light_col, args, light_intensity=None):
	cur_render, _ , _ = render_maps_outdataVec_from_mlp_out(
			L_bar, mlp_L, wi_array, wo_array, light_col, args,
			multi_wo=args.multi_wo, light_intensity=light_intensity)
	if args.neumip_tonemap:
		cur_render = render_utils.neuMIPTonemapper(cur_render)
	else:
		cur_render = render_utils.process_raw_render(cur_render)
	return cur_render


def get_render_maps_outDataVec(dataVec_list_in, model, L_bar, 
	wi, wo_array, light_col, args, multi_wo):

	model_output = process_full_img_data_in_mlp(model, *dataVec_list_in)

	render, maps, outDataVec = render_maps_outdataVec_from_mlp_out(
		L_bar, model_output, wi, wo_array, 
		light_col, args, multi_wo)

	return render, maps, outDataVec


def output_avg_maps(
	args, dataset, light_col, kernel_size, level_str):

	out_path = args.output_dir
	mat_id = args.train_out_id

	img_data = dataset.get_material_data(mat_id)

	maps = maps_from_data(img_data)
	maps = average_maps(maps, kernel_size)
	
	avg_render = render_utils.batch_render(maps, 
		renderer.Renderer.z_vector, renderer.Renderer.z_vector,
		light_col, args, multi_wo=False)

	name = dataset.get_material_name(mat_id)
	maps_name = dataset.get_output_maps_name()

	maps[0] = torch_norm(maps[0])

	for i in range(len(maps_name)):
		if maps_name[i] == "normal":
			file_utils.write_normal(
				out_path + "0" + name + "_" 
				+ maps_name[0] + "_" 
				+ level_str + "_avg.png",
				maps[i].cpu().numpy(), 
				args.opengl_normals,
				args.verbose)
		elif maps_name[i] == "albedo":
			file_utils.write_image_as_png(
				out_path + "0" + name + "_" 
				+ maps_name[i] + "_" 
				+ level_str + "_avg.png",
				render_utils.gammaCorrection(
					maps[i]).cpu().numpy(), args.verbose)
		else:
			file_utils.write_image_as_png(
				out_path + "0" + name + "_" 
				+ maps_name[i] + "_" 
				+ level_str + "_avg.png",
				maps[i].cpu().numpy(), args.verbose)
	
	avg_render = render_utils.process_raw_render(avg_render)
	file_utils.write_image_as_png(
		out_path + "0" + name + "_" 
		+ level_str + "_zrender_avg.png", 
		avg_render.cpu().numpy(), args.verbose)
	
	return


def compute_full_lod_gt_render(lod_lvl, L0, wi_vec, wo_vec, light_col, args):
	
	gt_render_lod = compute_L1_gt_render(L0, wi_vec, wo_vec, light_col, args)

	while lod_lvl > 1:
		gt_render_lod = average_patches(gt_render_lod, ks = 2)
		lod_lvl -= 1

	return gt_render_lod
