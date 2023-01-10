import torch
import cv2
import numpy as np
import os

from tqdm import tqdm

from mipnet.renderer import Renderer

from . import data_utils
from . import file_utils


def load_exr_render_from_folder(folder):
	images = []
	for filename in os.listdir(folder):
		if filename.endswith('.exr'):
			img = read_exr_render(os.path.join(folder, filename))
		if img is not None:
			images.append(img)
	return np.moveaxis(np.array(images), 0, -1), len(images)


def read_exr_render(img_path):
	image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	assert image is not None
	if image.ndim > 2:
		image = image[:, :, 0]  # keep only x channel
	return image


def load_roughness(images_path, mat_name):
	roughnessmap = file_utils.read_img_as_float(
		images_path + mat_name + "/roughness.png",
		multi_channel=False)
	# clamp the min roughness values to the min roughness (see renderer.py)
	min_roughness = 0.045
	roughnessmap[roughnessmap < min_roughness] = min_roughness
	return torch.from_numpy(roughnessmap).float()


def load_normal(images_path, mat_name, opengl_normals):
	normalmap = torch.from_numpy(file_utils.read_img_as_float(
		images_path + mat_name + "/normal.png",
		multi_channel=True))
	return unpack_normal(normalmap, opengl_normals).float()


def load_albedo(images_path, image_name):
	img = torch.from_numpy(file_utils.read_img_as_float(
		images_path + image_name + "/basecolor.png",
		True))  # multi-channel
	return file_utils.srgb2linrgb(img)


def load_metallic(images_path, image_name):
	img = file_utils.read_img_as_float(
		images_path + image_name + "/metallic.png",
		False)  # multi-channel
	return torch.from_numpy(img).float()


def load_height(images_path, image_name):
	img = file_utils.read_img_as_float(
		images_path + image_name + "/height.png",
		False)  # multi-channel
	return torch.from_numpy(img).float()


def unpack_normal(normals, opengl_normals):
	normals = 2.0 * normals - 1.0
	if opengl_normals:
		normals[..., 1] = -normals[..., 1]
	return normals


class MIPNetDataset(torch.utils.data.Dataset):

	"""
	A MIPNet dataset is organized as follows:

	root/
	+- data/
	|   +- materials_train/
	|   |   +- material001/
	|   |   |   +- baseColor.png
	|   |   |   +- height.png
	|   |   |   +- metallic.png
	|   |   |   +- normal.png
	|   |   |   +- roughness.png
	|   |   +- material002/
	|   |   |   +- baseColor.png
	|   |   |   +- height.png
	|   |   |   +- metallic.png
	|   |   |   +- normal.png
	|   |   |   +- roughness.png
	"""

	def load_single_material(self, mat_path, mat_folder, mat_id):
		
		normal = load_normal(mat_path, mat_folder, self.opengl_normals)
		if normal.shape[0] != self.base_res:
			file_utils.exit_with_message(
				"inconsistency of base res: "
				+ "normal.shape[0] != args.map_size")
		roughness = load_roughness(mat_path, mat_folder)
		albedo = load_albedo(mat_path, mat_folder)
		metallic = load_metallic(mat_path, mat_folder)
		height = load_height(mat_path, mat_folder) - 0.5

		# building the (alpha_t, 0, 0, alpha_b) matrix
		# alpha = roughness² is the perceptually linear roughness which is squared later during shading
		alpha = torch.clamp(torch.pow(roughness, 2.0), min=Renderer.min_alpha, max=1.0)
		self.data[0][mat_id, :, :, 0:2] = normal[..., :2]
		self.data[0][mat_id, :, :, 2] 	= alpha
		self.data[0][mat_id, :, :, 3] 	= alpha
		#self.data[0][mat_id, :, :, 4] 	= 0
		self.data[0][mat_id, :, :, 5] 	= self.disp_val * height
		self.data[0][mat_id, :, :, 6:9] = albedo
		self.data[0][mat_id, :, :, 9] 	= metallic
		self.data[0][mat_id, :, :, 10:] = self.position
		
		
	def list_all_paths(self, mat_list_file):
		if os.path.exists(mat_list_file):
			with open(mat_list_file, 'r') as f:
				all_materials = f.read()
			return all_materials.split('\n')
		else:
			return []
        

	def load_materials_from_folder(self, mat_path, file_to_load, verbose=True):
		txt_mat_names = self.list_all_paths(file_to_load)
		# self.nb_mat = len(txt_mat_names)
		if verbose:
			print("found: " + str(len(txt_mat_names)) + " in txt file")
		self.nb_mat = 0
		self.mat_names = []
		for mat_folder in txt_mat_names:
			if os.path.exists(mat_path + mat_folder):
				self.mat_names.append(mat_folder)
			else:
				print("could not find: " + mat_path + mat_folder)
		self.nb_mat = len(self.mat_names)
		if verbose:
			print("found: " + str(self.nb_mat) + " in folder")
		prog_bar = tqdm(total=self.nb_mat, leave=False)
		if self.nb_mat == 0 and verbose:
			print("no materials found in dir: " + mat_path)
		if len(self.data) == 0:
			self.data = [torch.zeros((self.nb_mat, self.base_res, 
				self.base_res, self.data_dims))]
		for mat_id, mat_folder in enumerate(self.mat_names):
			self.load_single_material(mat_path, mat_folder, mat_id)
			prog_bar.update(1)
		prog_bar.close()


	def compute_nb_of_tiles(self, lod_idx):
		return int((self.base_res // self.tile_width) ** 2 \
			* (4**lod_idx - 1) // (4**(lod_idx - 1) * 3))


	def get_data_lod_lvl(self, tile_idx):
		#start with lod 0, 1, 2 ...
		lod_lvl = 1
		while lod_lvl < self.nb_lod:
			if tile_idx < self.compute_nb_of_tiles(lod_lvl):
				return lod_lvl - 1
			else:
				lod_lvl += 1
		return lod_lvl - 1


	def __init__(self, images_path, file_to_load, args, device, verbose=True):

		self.base_res = args.map_size
		self.data = []
		self.mlp_maps = None
		self.nb_lod = None

		self.img_path 		= images_path
		self.disp_val 		= args.disp_val
		self.opengl_normals = args.opengl_normals
		self.device 		= device

		self.tile_width		= 2 ** args.nb_levels
		
		# whole data tensor size
		# normals/anisoMatrix/height/albedo/metallic/position
		self.data_dims = 2 + 3 + 1 + 3 + 1 + 2 

		# create XY position maps
		self.position = data_utils.compute_2D_position_grid(self.base_res)

		self.load_materials_from_folder(images_path, file_to_load, verbose)

		self.map_out_name = [
			"normal", 
			"anisoMat",
			"height",
			"albedo", 
			"metallic"]

		self.nb_tiles = (self.base_res // self.tile_width) ** 2

		if self.nb_tiles < 1:
			file_utils.exit_with_message("wrong number of levels given")


	def __getitem__(self, index):

		mat_idx  = index // self.nb_tiles
		tile_idx = index %  self.nb_tiles

		tile_res = self.base_res // self.tile_width

		tile_x = tile_idx % tile_res
		tile_y = tile_idx // tile_res

		data_x = self.tile_width * tile_x
		data_y = self.tile_width * tile_y

		X = self.data[0][mat_idx,
			data_y: data_y + self.tile_width,
			data_x: data_x + self.tile_width]
		
		return X


	def __len__(self):
		return self.nb_tiles * self.nb_mat

	def get_base_res(self):
		return self.base_res

	def get_nb_of_materials(self):
		return self.nb_mat

	def get_tile_width(self):
		return self.tile_width

	def get_nb_tiles(self):
		return self.nb_tiles

	def get_output_maps_name(self):
		return self.map_out_name

	def get_nb_of_output_maps(self):
		return len(self.map_out_name)

	def get_material_list(self):
		return self.mat_names
			
	def get_material_data(self, mat_idx):
		if mat_idx > self.nb_mat - 1:
			file_utils.exit_with_message(
				"wrong material index provided")
		return self.data[0][mat_idx, ...].to(self.device)

	def get_material_name(self, mat_idx):
		if mat_idx > self.nb_mat - 1:
			file_utils.exit_with_message(
				"wrong material index provided")
		return self.mat_names[mat_idx]

	def get_mlp_out_test_pyr(self, mat_idx, lod):
		return self._mlp_out_test_pyr[mat_idx][lod]

	def set_mlp_out_test_pyr(self, mlp_maps_pyr):
		self._mlp_out_test_pyr = mlp_maps_pyr