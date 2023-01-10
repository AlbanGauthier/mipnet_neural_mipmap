import torch
import numpy as np
from flip_pytorch import flip_loss
from tqdm import tqdm

from . import render_utils
from . import data_utils


def multiscale_ref_render(
	img_data, wi_vec, wo_array, light_col, args, multi_wo):

	maps = data_utils.maps_from_data(img_data)
	gt_render = render_utils.batch_render(
		maps, wi_vec, wo_array, light_col, args, multi_wo)
	N = gt_render.shape[0]
	render_pyramid = []
	while N // 2 >= args.render_res_thresh:
		gt_render = data_utils.average_patches(
			gt_render, ks=2)
		srgb_render = \
			render_utils.process_raw_render(gt_render)
		render_pyramid.append(srgb_render.cpu())
		N = gt_render.shape[0]

	return render_pyramid


def multiscale_avg_render(
	L0, wi_vec, wo_array, light_col, nb_maps, args, multi_wo):

	render_pyramid = []
	maps_pyr = [[] for _ in np.arange(nb_maps)]

	maps = data_utils.maps_from_data(L0)
	for i in range(nb_maps):
		maps_pyr[i].append(torch.clone(maps[i]))

	N = maps[0].shape[0]

	while N > 1:

		maps = data_utils.average_maps(maps, ks=2)
		for i in range(nb_maps):
			maps_pyr[i].append(torch.clone(maps[i]))

		if N // 2 >= args.render_res_thresh:
			avg_render = render_utils.batch_render(
				maps, wi_vec, wo_array,
				light_col, args, multi_wo)
			avg_render = render_utils.process_raw_render(avg_render)
			render_pyramid.append(avg_render.cpu())

		N = maps[0].shape[0]
	
	return render_pyramid, maps_pyr


def inference_biscale(model_A, model_B, L0, wi_vec, wo_array, 
	light_col, nb_maps, args, multi_wo):

	# 1024x1024
	H = L0.shape[0]
	downsample = data_utils.average_patches

	render_pyramid = []
	## list of [ normal, roughness, albedo, metallic ]
	maps_pyr = [[] for _ in np.arange(nb_maps)]

	maps = data_utils.maps_from_data(L0)
	maps = data_utils.get_renorm_clamped_maps(maps)
	for i in range(nb_maps):
		maps_pyr[i].append(maps[i].clone())

	# get L1 from A(L0) 512x512
	L1_bar = downsample(L0)
	render, mlp_maps, L1 = data_utils.get_render_maps_outDataVec(
		[L0], model_A, L1_bar, wi_vec, wo_array, light_col, args, multi_wo)
	mlp_maps = data_utils.get_renorm_clamped_maps(mlp_maps)
	for i in range(nb_maps):
		maps_pyr[i].append(mlp_maps[i].clone())
	render = render_utils.process_raw_render(render)
	render_pyramid.append(render.cpu())

	# init to get L2: 256x256
	L_i = L0
	L_ip1 = L1
	mlp_L_ip2_bar = downsample(L1)
	ker_thresh = 4

	while H // ker_thresh >= 1:

		_, _, L_ip2 = data_utils.get_render_maps_outDataVec(
			[L_ip1], model_A, mlp_L_ip2_bar, wi_vec, 
			wo_array, light_col, args, multi_wo)
		
		dataVec_list = [L_i, L_ip1, L_ip2]

		render, mlp_maps, L_ip2 = data_utils.get_render_maps_outDataVec(
			dataVec_list, model_B, mlp_L_ip2_bar,
			wi_vec, wo_array, light_col, args, multi_wo)

		mlp_maps = data_utils.get_renorm_clamped_maps(mlp_maps)
		for i in range(nb_maps):
			maps_pyr[i].append(mlp_maps[i].clone())

		if H // 4 >= args.render_res_thresh:
			render = render_utils.process_raw_render(render)
			render_pyramid.append(render.cpu())

		#next step preparation
		L_i 	= L_ip1
		H = L_i.shape[0]

		if H // ker_thresh >= 1:
			L_ip1 	= L_ip2
			mlp_L_ip2_bar = downsample(L_ip1)


	return render_pyramid, maps_pyr


def custom_FLIP(test, ref):
	# Transform reference and test to opponent color space
	reference_opponent = \
		flip_loss.color_space_transform(ref, 'srgb2ycxcz')
	test_opponent = \
		flip_loss.color_space_transform(test, 'srgb2ycxcz')
	qc = 0.7; qf = 0.5; pc = 0.4; pt = 0.95; eps = 1e-15
	pixels_per_degree = (0.7 * 3840 / 0.7) * np.pi / 180
	deltaE = flip_loss.compute_ldrflip(
		test_opponent, reference_opponent, 
		pixels_per_degree, qc, qf, pc, pt, eps)

	return deltaE


def compute_FLIP_pyr_img_pytorch(
	pyr_gt_render, pyr_test_render, multi_wo):
	flip_pyr = []
	for level, gt_render in enumerate(pyr_gt_render):
		if gt_render.shape[0] >= 2048:
			flip_tmp = process_FLIP_using_tiles(
				pyr_test_render[level].permute(3, 2, 0, 1), 
				gt_render.permute(3, 2, 0, 1))
		else:
			flip_tmp = custom_FLIP(
				pyr_test_render[level].permute(3, 2, 0, 1), 
				gt_render.permute(3, 2, 0, 1))
		flip_tmp = flip_tmp.squeeze().cpu().numpy()
		if multi_wo:
			pass
		flip_pyr.append(flip_tmp)
	return flip_pyr


def process_FLIP_using_tiles(test, ref):
	L, _, H, W = test.shape
	tile_size = H // 1024
	tile_height = H // tile_size
	tile_width = W // tile_size
	flip_out = torch.zeros((L, 1, H, W))
	prog_bar = tqdm(total=tile_size*tile_size, leave=False)
	for i in range(tile_size):
		for j in range(tile_size):
			flip_out[:, :, 
				i * tile_height: (i + 1) * tile_height,
				j * tile_width : (j + 1) * tile_width] = \
					custom_FLIP(
						test[:, :,
				i * tile_height: (i + 1) * tile_height,
				j * tile_width : (j + 1) * tile_width], 
						ref[:, :,
				i * tile_height: (i + 1) * tile_height,
				j * tile_width : (j + 1) * tile_width])
			prog_bar.update(1)
	prog_bar.close()
	return flip_out