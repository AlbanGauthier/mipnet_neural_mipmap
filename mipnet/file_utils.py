import torch
import cv2
import json, os, time, re
import numpy as np
import matplotlib.pyplot as plt
from subprocess import PIPE, run
from argparse import Namespace

from . import data_utils
from . import render_utils
from . import renderer

def out(command):
    result = run(
		command, stdout=PIPE, stderr=PIPE, 
		universal_newlines=True, shell=True)
    return result.stdout


def exit_with_message(message):
	print("##############################################")
	print(message)
	print("##############################################")
	exit(0)


def get_training_time(args):
	t = time.localtime()
	res_str = time.strftime("%b%d_%Hh%M", t)
	return res_str


def get_training_id(args):
	t = time.localtime()
	current_time = time.strftime("%b%d_%Hh%M", t)
	res_str = ""
	res_str += current_time
	res_str += "_lvl" + str(args.nb_levels)
	res_str += "_bs" + str(args.batch_size)
	res_str += "_" + str(args.hid_nb_A) + "-" + str(args.hid_size_A)
	res_str += "_" + str(args.hid_nb_B) + "-" + str(args.hid_size_B)
	res_str += "_" + str(args.learning_rate)
	res_str += "/"
	return res_str


def handle_empty_outdir(args):
	## if output_dir is empty and training
	if args.mode == "train":
		if not args.output_dir:
			args.output_dir = "./results/" + get_training_id(args)
		elif args.timestamp:
			args.output_dir += get_training_id(args)
	## else if empty and eval/test
	elif not args.output_dir:
		args.output_dir = "./results/" + args.mode + "/"
	elif args.output_dir == "./results/":
		args.output_dir += args.mode + "/"
	return args


def get_dict_from_model(model):
	# works only for model defined with self.func = Sequential(...)
	modules = model.named_modules()
	ret = ""
	for m in modules:
		if m[0] == 'mlp':
			ret = repr(m[1])
			return ret
	return "__error__ in get_dict_from_model()"


def save_training_metadata(args, model_A, model_B, train_dataset, val_dataset):
	
	json_dict = {}

	git_id = out("git rev-parse HEAD")
	json_dict["git_id"] = git_id
	json_dict["training_time"] = get_training_time(args)
	
	model_list = get_dict_from_model(model_A).split("\n")
	json_dict["architectureA"] = {}
	for i, elem in enumerate(model_list):
		json_dict["architectureA"][str(i)] = elem

	model_list = get_dict_from_model(model_B).split("\n")
	json_dict["architectureB"] = {}
	for i, elem in enumerate(model_list):
		json_dict["architectureB"][str(i)] = elem
	
	json_dict["mat_train"] = {}
	mat_list = train_dataset.get_material_list()
	for i, elem in enumerate(mat_list):
		json_dict["mat_train"][str(i)] = elem

	json_dict["mat_val"] = {}
	if val_dataset is not train_dataset and val_dataset is not None:
		mat_list = val_dataset.get_material_list()
		for i, elem in enumerate(mat_list):
			json_dict["mat_val"][str(i)] = elem

	json_dict["mat_eval"] = {}
	if args.eval_file != args.train_file:
		if os.path.exists(args.eval_file):
			with open(args.eval_file, 'r') as f:
				material_list = f.read()
				material_list = material_list.split('\n')
			for i, elem in enumerate(material_list):
				json_dict["mat_eval"][str(i)] = elem
		else:
			print("eval file does not exist")

	json_dict["torch_seed"] = torch.initial_seed()
	json_dict.update(vars(args))
	json_dict["total time"] = 0
	json_dict["total loss values"] = 0

	if args.restart:
		json_dict["train_loss"] = args.train_loss
		json_dict["valid_loss"] = args.valid_loss
	else:
		json_dict["train_loss"] = {}
		json_dict["valid_loss"] = {}
	
	jsonString = json.dumps(json_dict, indent=2)
	jsonFile = open(args.output_dir + "data.json", "w")
	jsonFile.write(jsonString)
	jsonFile.close()


def update_loss_json(epoch_id, loss_list, json_dir):
	with open(json_dir + "data.json", "r+") as jsonFile:
		# Load existing data into a dict.
		file_data = json.load(jsonFile)
		# Sets number of train values
		file_data["total loss values"] = epoch_id
		# Join new_loss with file_data
		file_data["train_loss"]["loss " + str(epoch_id)] = loss_list[0]
		file_data["valid_loss"]["loss " + str(epoch_id)] = loss_list[1]
		# Sets file's current position at offset.
		jsonFile.seek(0)
		# convert back to json.
		json.dump(file_data, jsonFile, indent=2)


def update_json_with_data(dict, json_dir):
	with open(json_dir + "data.json", "r+") as jsonFile:
		file_data = json.load(jsonFile)
		for key in dict:
			file_data[key] = dict[key]
		jsonFile.seek(0)
		json.dump(file_data, jsonFile, indent=2)


def save_models(modelA, modelB, epoch, json_dir):
	torch.save(modelA.state_dict(), json_dir + \
		"epoch_{}_mipnet_A.model".format(epoch))
	torch.save(modelB.state_dict(), json_dir + \
		"epoch_{}_mipnet_B.model".format(epoch))
	print("Checkpoint saved")


def output_loss_graph(args, model_path, hasValidationData):
	with open(model_path + "data.json", "r+") as jsonFile:
		# Load existing data into a dict.
		json_dict = json.load(jsonFile)

		trainloss_vec = []
		validloss_vec = []
		total_train_val = json_dict["total loss values"]
		for i in range(total_train_val):
			trainloss_vec.append(json_dict["train_loss"].get("loss " + str(i+1), 0))
			validloss_vec.append(json_dict["valid_loss"].get("loss " + str(i+1), 0))
		plt.plot(np.arange(1, total_train_val+1), 
			trainloss_vec, c='blue', label="train loss")
		if hasValidationData:
			plt.plot(np.arange(1, total_train_val+1), 
				validloss_vec, c='green', label="validation loss")
		
		# plt.ylim((0.0140, 0.0160))
		# plt.xlim((-1, 50))

		# Create empty plot with blank marker containing the extra labels
		plt.plot([], [], ' ', c='black', label="ID "
			+ str(json_dict["training_time"]))
		plt.plot([], [], ' ', c='black', label="number of epochs   "
			+ str(json_dict["num_epochs"]))
		plt.plot([], [], ' ', c='black', label="torch seed         "
			+ str(json_dict["torch_seed"]))
		
		plt.plot([], [], ' ', c='black', label="batch size         "
			+ str(json_dict["batch_size"]))
		plt.plot([], [], ' ', c='black', label="hid layer size A   "
			+ str(json_dict["hid_size_A"])) 
		plt.plot([], [], ' ', c='black', label="hid layer nb   A   "
			+ str(json_dict["hid_nb_A"]))   
		plt.plot([], [], ' ', c='black', label="hid layer size B   "
			+ str(json_dict["hid_size_B"])) 
		plt.plot([], [], ' ', c='black', label="hid layer nb   B   "
			+ str(json_dict["hid_nb_B"]))  
		plt.plot([], [], ' ', c='black', label="train wi dir       "
			+ str(json_dict["nb_wi_train"]))
		plt.plot([], [], ' ', c='black', label="train wo dir       "
			+ str(json_dict["nb_wo_train"]))
		plt.plot([], [], ' ', c='black', label="learning rate      "
			+ str(json_dict["learning_rate"])) 
		
		plt.plot([], [], ' ', c='black', label="light sampling     "
            		+ json_dict["train_light_sampling"])
		plt.plot([], [], ' ', c='black', label="loss type          "
            		+ json_dict["loss_type"])
		plt.plot([], [], ' ', c='black', label="non linearity      "
					+ json_dict["non_linearity"])
		
		plt.plot([], [], ' ', c='black', label="shuffle            "
			+ str(json_dict["shuffle"]))   
		plt.plot([], [], ' ', c='black', label="multi wo           "
			+ str(json_dict["multi_wo"]))
		
		plt.plot([], [], ' ', c='black', label="neumip tonemap     "
			+ str(json_dict["neumip_tonemap"]))
		plt.plot([], [], ' ', c='black', label="colored render     "
			+ str(json_dict["colored_render"]))

		plt.plot([], [], ' ', c='black', label="training levels    "
			+ str(json_dict["nb_levels"]))
	
		plt.plot([], [], ' ', c='black', label="total time         "
			+ str(json_dict["total time"]))
		
		plt.legend(
			prop={'family': 'DejaVu Sans Mono', 'size': 6}, 
			bbox_to_anchor=(1.0, 1.0),
			labelcolor='linecolor',
			ncol = 2)

		t = time.localtime()
		current_time = time.strftime("%b%d_%Hh%M", t)
		
		if args.output_pdf:
			plt.savefig(model_path + "0loss_plt_" + current_time + ".pdf", 
				bbox_inches='tight')

		plt.savefig(model_path + "0loss_plt_" + current_time + ".png", 
			bbox_inches='tight', dpi=200)
		plt.close()
		print("output loss graph in: " + model_path)


def process_image_as_float(img, multi_channel=True):
    img_type = img.dtype
    if multi_channel:  # converting BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (img_type == np.dtype('uint16')):
        img = img / float(pow(2, 16) - 1)
    elif (img_type == np.dtype('uint8')):
        img = img / float(pow(2, 8) - 1)
    return img


def read_img_as_float(img_path, multi_channel=True):
	image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	if image is None:
		exit_with_message(
			"No file at specified path: " + img_path)
	#roughness sometimes has 3 channels
	if not multi_channel and image.ndim > 2:
		image = image[:, :, 0]
	return process_image_as_float(image, multi_channel)


def srgb2linrgb(input_color):
    limit = 0.04045
    transformed_color = torch.where(
        input_color > limit,
        torch.pow((torch.clamp(input_color, min=limit) + 0.055) / 1.055, 2.4),
        input_color / 12.92
    )  # clamp to stabilize training
    return transformed_color


def output_aniso_matrix(name, pyr_list, maps_names, verbose):
	
	for map_idx, map_name in enumerate(maps_names):
		if map_name == "albedo" \
		or map_name == "metallic" \
		or map_name == "height" \
		or map_name == "normal":
			continue
		elif map_name == "roughness_x":
			r_t = pyr_img_from_list(
				pyr_list[map_idx], numpy_arrays = False).cpu().numpy()
		elif map_name == "roughness_y":
			r_b = pyr_img_from_list(
				pyr_list[map_idx], numpy_arrays = False).cpu().numpy()
		elif map_name == "anisoAngle":
			angle = pyr_img_from_list(
				pyr_list[map_idx], numpy_arrays = False).cpu().numpy()
		else:
			exit_with_message("error in the map name:", map_name)
	
	w, h, _ = angle.shape
	matrix_img = np.zeros((w, h, 3), dtype=np.float32)

	# building the (at², 0, 0, ab²) matrix
	r_t_sqr = np.power(r_t[..., 0], 2.0) 
	# r_t² is the non perceptual roughness 
	# which is squared later in the shader
	r_b_sqr = np.power(r_b[..., 0], 2.0)
	
	cos_angle = np.cos(angle[..., 0])
	cos_sqr = cos_angle*cos_angle

	sin_angle = np.sin(angle[..., 0])
	sin_sqr = sin_angle*sin_angle

	matrix_img[..., 0] = r_b_sqr * cos_sqr + r_t_sqr * sin_sqr
	matrix_img[..., 1] = r_b_sqr * sin_sqr + r_t_sqr * cos_sqr
	matrix_img[..., 2] = (r_t_sqr - r_b_sqr) * cos_angle * sin_angle #TODO: verify
	matrix_img[..., 2] = 0.5 * matrix_img[..., 2] + 0.5

	write_image_as_exr(name, matrix_img)
	write_image_as_png(name, matrix_img, verbose)

	return


def output_image_pyr(name, pyr_list, map_name, opengl_normals, verbose):
	if map_name == "height":
		return
	if map_name == "normal":
		for i in range(len(pyr_list)):
			if opengl_normals:
				pyr_list[i][..., 1] = -pyr_list[i][..., 1]
			pyr_list[i] = 0.5 * pyr_list[i] + 0.5
	if map_name == "albedo":
		for i in range(len(pyr_list)):
			pyr_list[i] = render_utils.gammaCorrection(pyr_list[i])
	if map_name == "anisoMat":
		for i in range(len(pyr_list)):
			pyr_list[i][..., 0] = torch.clamp(pyr_list[i][..., 0], 
				min=renderer.Renderer.min_alpha, max=1.0)
			pyr_list[i][..., 1] = torch.clamp(pyr_list[i][..., 1], 
				min=renderer.Renderer.min_alpha, max=1.0)
			pyr_list[i][..., 2] = 0.5 * pyr_list[i][..., 2] + 0.5
	pyr_img = pyr_img_from_list(
		pyr_list, numpy_arrays = False).cpu().numpy()
	write_image_as_png(name, pyr_img, verbose)
	return


def pyr_img_from_list_results(img_list, numpy_arrays):
	height = img_list[0].shape[0]
	width = img_list[0].shape[0] + img_list[1].shape[0]
	if img_list[0].ndim == 2:
		if numpy_arrays:
			pyr_img = np.zeros((height, width))
		else:
			pyr_img = torch.zeros((height, width))
	else:
		channels = img_list[0].shape[2]
		if numpy_arrays:
			pyr_img = np.zeros((height, width, channels))
		else:
			pyr_img = torch.zeros((height, width, channels))
	
	H = img_list[0].shape[0]
	if img_list[0].ndim == 2:
		pyr_img[:H, :H] = img_list[0]
		pyr_img[:H//2, H:] = img_list[1]
	else:
		pyr_img[:H, :H, :] = img_list[0]
		pyr_img[:H//2, H:, :] = img_list[1]
	
	voffset = H // 2
	for i in range(2, len(img_list)):
		img = img_list[i]
		if img_list[0].ndim == 2:
			w, h = img.shape
			pyr_img[voffset:voffset+h, -w:] = img
		else:
			w, h, _ = img.shape
			pyr_img[voffset:voffset+h, -w:] = img
		voffset += h
	return pyr_img


def pyr_img_from_list(img_list, numpy_arrays):
	height = img_list[0].shape[0]
	width = 0
	for img in img_list:
		width += img.shape[0]
	if img_list[0].ndim == 2:
		if numpy_arrays:
			pyr_img = np.zeros((height, width))
		else:
			pyr_img = torch.zeros((height, width))
	else:
		channels = img_list[0].shape[2]
		if numpy_arrays:
			pyr_img = np.zeros((height, width, channels))
		else:
			pyr_img = torch.zeros((height, width, channels))
	offset = 0
	for img in img_list:
		if img_list[0].ndim == 2:
			w, h = img.shape
			pyr_img[:h, offset:offset+w] = img
		else:
			w, h, _ = img.shape
			pyr_img[:h, offset:offset+w, :] = img
		offset += w
	return pyr_img


def write_normal(path, array, opengl_normals, verbose=False):
	if opengl_normals:
		array[..., 1] = -array[..., 1]
	array = np.array(65535 * (0.5 * array + 0.5), dtype=np.uint16)
	array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
	if not cv2.imwrite(path, array):
		print("error saving normal map at:", path)
		assert False
	if verbose:
		print("saved normal map: " + path)


def write_image_as_png(path, vec, verbose=False):
	assert path.endswith(".png")
	array = np.clip(vec, 0.0, 1.0)
	array = np.array(65535 * array, dtype=np.uint16)
	if array.ndim==3:  # converting BGR to RGB
		if array.shape[2] == 4:
			array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
		elif array.shape[2] == 3:
			array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
	if not cv2.imwrite(path, array):
		print("error saving img at:", path)
		assert False
	if verbose:
		print("saved image in: " + path)


def write_image_as_exr(path, vec):
	if path.endswith(".png"):
		path = path[:-4]
	if not path.endswith(".exr"):
		path += ".exr"
	if vec.ndim==3:
		# pad with one channel of zeros
		if vec.shape[2] == 2:
			vec = np.pad(vec, ((0, 0), (0, 0), (0, 1)))
		# converting RGB to BGR
		vec = cv2.cvtColor(vec, cv2.COLOR_RGB2BGR)
	if cv2.imwrite(path, vec):
		print("saved image in: " + path)
	else:
		print("error saving img")
		assert False


def load_args_from_json(args):

	if os.path.exists(args.model_path + "data.json"):
		with open(args.model_path + "data.json", "r+") as jsonFile:
			json_dict = json.load(jsonFile)

			args.hid_size_A 		= json_dict["hid_size_A"]
			args.hid_nb_A 			= json_dict["hid_nb_A"]
			args.hid_size_B 		= json_dict["hid_size_B"]
			args.hid_nb_B 			= json_dict["hid_nb_B"]
			args.renderer_type 		= json_dict["renderer_type"]

			if args.restart:
				args.train_loss = json_dict["train_loss"]
				if "valid_loss" in json_dict:
					args.valid_loss 	= json_dict["valid_loss"]
	else:
		exit_with_message("could not find: " + args.model_path + "data.json")

	return args


def load_model_args_from_json(model_path):
	
	if os.path.exists(model_path + "data.json"):
		with open(model_path + "data.json", "r+") as jsonFile:
			json_dict = json.load(jsonFile)
			new_dict = {k:json_dict[k] for k in json_dict if not re.match('loss epoch', k)}
			args = Namespace(**new_dict)
	else:
		exit_with_message("could not find: " + model_path + "data.json")

	return args