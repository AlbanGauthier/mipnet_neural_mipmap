import torch
import numpy as np
import os
from tqdm import tqdm

from .renderer import Renderer
from .dataLoader import MIPNetDataset
from . import data_utils
from . import test_utils
from . import file_utils
from . import render_utils

def write_output_maps(
	out_dir, maps, args, render, mat_name, 
	maps_name, model_idx, level_str):

	if args.output_train_maps:
		
		file_utils.write_normal(
			out_dir + mat_name + "out_normal"
			+ "_lod" + level_str + "_" 
			+ str(model_idx).zfill(3) + ".png",
			maps[0].cpu().numpy(), 
			args.opengl_normals,
			args.verbose)

		for i in range(1, len(maps_name)):
			
			if maps_name[i] == "metallic":
				continue
			if maps_name[i] == "albedo":
				continue
			else:
				file_utils.write_image_as_png(
					out_dir + mat_name + "out_" + maps_name[i] 
					+ "_lod" + level_str + "_" 
					+ str(model_idx).zfill(3) + ".png",
					maps[i].cpu().numpy(), args.verbose)

	if args.output_train_render:

		render = render_utils.process_raw_render(render)
		file_utils.write_image_as_png(
			out_dir + mat_name + "out_zrender_NN"
			+ "_lod" + level_str + "_"
			+ str(model_idx).zfill(3) + ".png",
			render.cpu().numpy(), args.verbose)

	return


def eval_output_maps_and_render(
	model_A, model_B, dataset, args, model_idx):

	out_dir = args.output_dir

	model_A.eval()
	model_B.eval()

	L0 = dataset.get_material_data(args.train_out_id)
	mat_name = dataset.get_material_name(args.train_out_id)
	nb_maps = dataset.get_nb_of_output_maps()
	maps_name = dataset.get_output_maps_name()

	## Composition of MLP over LoD_0
	mlp_render_pyr, mlp_maps_pyr = test_utils.inference_biscale(
			model_A, model_B, L0, 
			Renderer.z_vector,
			Renderer.z_vector,
			Renderer.white_light, 
			nb_maps, args, multi_wo=False)
	
	if args.output_train_render:
		file_utils.output_image_pyr(
			out_dir + mat_name + "out_zrender_NN_" + str(model_idx).zfill(3) + ".png",
			mlp_render_pyr, map_name="render", opengl_normals=args.opengl_normals,
			verbose=args.verbose)

	if args.output_train_maps:
		for map_idx, map_name in enumerate(maps_name):
			file_utils.output_image_pyr(
				out_dir + mat_name + "out_" + map_name + str(model_idx).zfill(3) + ".png", 
				mlp_maps_pyr[map_idx], map_name, args.opengl_normals, args.verbose)

	return


def eval_model_output_mips(args, model_A, model_B, device, isTrain):

	print("evaluating model on", "training" if isTrain else "evaluation", "data")
	material_filename = args.train_file if isTrain else args.eval_file

	if os.path.exists(material_filename):
		with open(material_filename, 'r') as f:
			material_list = f.read()
			material_list = material_list.split('\n')
			nb_mat = len(material_list)
	else:
		print("could not find: ", material_filename)
		return

	if args.mode == "eval":
		model_A.load_state_dict(torch.load(
			args.model_path + "epoch_" +
			args.model_index + "_mipnet_A.model"))
		model_B.load_state_dict(torch.load(
			args.model_path + "epoch_" + \
			args.model_index + "_mipnet_B.model"))

	model_A.eval()
	model_B.eval()

	prog_bar = tqdm(total=nb_mat, leave=False)
	with torch.no_grad():
		for mat_idx in range(nb_mat):
			
			with open("single_mat_eval.txt", 'w') as f:
				f.write(material_list[mat_idx])

			#create single material dataset
			single_mat_dataset = MIPNetDataset(
				args.data_path, "single_mat_eval.txt", args, device, verbose=False)

			mat_name = single_mat_dataset.get_material_name(0)
			img_data = single_mat_dataset.get_material_data(0)
			maps_name = single_mat_dataset.get_output_maps_name()
			nb_maps = single_mat_dataset.get_nb_of_output_maps()

			evaluate_multiscale_model(model_A, model_B, img_data, mat_name, 
				args, nb_maps, maps_name, isTrain)

			if args.verbose:
				print("processed for evaluation:", mat_name)
			prog_bar.update(1)
	prog_bar.close()

	if os.path.exists("single_mat_eval.txt"):
		os.remove("single_mat_eval.txt")

	return


def compute_flip_error_images(out_dir, mat_name):
	
	# FLIP error maps
	os.system("python ./flip/flip.py -r "
			+ "\"" + out_dir + mat_name + "mip_zrender_ref.png" + "\"" + " -t "
			+ "\"" + out_dir + mat_name + "mip_zrender_mlp.png" + "\"" + " "
			+ "\"" + out_dir + mat_name + "mip_zrender_avg.png" + "\"" + " "
			+ "-d " + out_dir + " -v 0")

	# Rename flip files (and delete existing if any)
	out_flip_avg = out_dir + mat_name + "mip_flip_zrender_avg.png"
	if os.path.exists(out_flip_avg):
		os.remove(out_flip_avg)
	os.rename(out_dir +"flip." + "mip_zrender_ref" + "."
			+ "mip_zrender_avg" + ".67ppd.ldr" + ".png",
			out_flip_avg)
	
	out_flip_mlp = out_dir + mat_name + "mip_flip_zrender_mlp.png"
	if os.path.exists(out_flip_mlp):
		os.remove(out_flip_mlp)
	os.rename(out_dir + "flip." + "mip_zrender_ref" + "."
			+ "mip_zrender_mlp" + ".67ppd.ldr" + ".png",
			out_flip_mlp)

	return


def evaluate_multiscale_model(model_A, model_B, img_data, mat_name, 
	args, nb_maps, maps_names, isTrain):

	out_dir = args.output_dir
	if isTrain:
		out_dir += "train_render/"
	else:
		out_dir += "eval_render/"

	if not os.path.exists(out_dir+mat_name):
		os.makedirs(out_dir+mat_name)

	##
	## Reference render
	##
	ref_render_pyr = test_utils.multiscale_ref_render(
			img_data, Renderer.z_vector, Renderer.z_vector,
			Renderer.white_light, args, multi_wo=False)
	file_utils.output_image_pyr(
		out_dir + mat_name + "mip_zrender_ref.png",
		ref_render_pyr, map_name="render", 
		opengl_normals=args.opengl_normals, verbose=args.verbose)

	##
	## Averaged maps and render
	##
	avg_render_pyr, avg_maps_pyr = test_utils.multiscale_avg_render(
		img_data, Renderer.z_vector, Renderer.z_vector, Renderer.white_light, 
		nb_maps, args, multi_wo=False)
	print(out_dir + mat_name)
	file_utils.output_image_pyr(
		out_dir + mat_name + "mip_zrender_avg.png",
		avg_render_pyr, map_name="render", 
		opengl_normals=args.opengl_normals, verbose=args.verbose)
	
	if args.output_maps_pyr:
		for map_idx, map_name in enumerate(maps_names):
			if map_name == "albedo" or map_name == "metallic":
				continue
			file_utils.output_image_pyr(
				out_dir + mat_name + "mip_" + map_name + "_avg.png", 
				avg_maps_pyr[map_idx], map_name, args.opengl_normals, args.verbose)

	##
	## Mipnet maps and render
	##
	mlp_render_pyr, mlp_maps_pyr = test_utils.inference_biscale(
			model_A, model_B, img_data, 
			Renderer.z_vector, Renderer.z_vector, Renderer.white_light, 
			nb_maps, args, multi_wo=False)
	
	file_utils.output_image_pyr(
		out_dir + mat_name + "mip_zrender_mlp.png",
		mlp_render_pyr, map_name="render", 
		opengl_normals=args.opengl_normals, verbose=args.verbose)
	
	if args.output_maps_pyr:
		for map_idx, map_name in enumerate(maps_names):
			if map_name == "albedo" or map_name == "metallic":
				continue
			file_utils.output_image_pyr(
				out_dir + mat_name + "mip_" + map_name + "_mlp.png", 
				mlp_maps_pyr[map_idx], map_name, args.opengl_normals, args.verbose)

	if args.eval_compute_flip:
		compute_flip_error_images(out_dir, mat_name)

	return