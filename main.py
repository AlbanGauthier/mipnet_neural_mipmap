import os
import torch
import argparse
import matplotlib
import numpy as np

from mipnet.dataLoader 	import MIPNetDataset
from mipnet.renderer 	import Renderer
from mipnet.model 		import FullyConnected
import mipnet.file_utils 	as file_utils
import mipnet.training 		as training
import mipnet.testing 		as testing

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

############################
## Device is defined here ##
############################

device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

matplotlib_debug = False

def main(args):

	print("## Starting in mode: " + args.mode)
	print("###########################")

	np.random.seed(args.manual_seed)
	torch.manual_seed(args.manual_seed)

	if not matplotlib_debug:
		matplotlib.use("Agg")
	else:
		print("########################")
		print("### matplotlib_debug ###")
		print("########################")

	print('Using device:', device)

	if device.type == 'cuda':
		print(torch.cuda.get_device_name(0))
		print("")

	if args.mode == "eval" or args.restart:
		args = file_utils.load_args_from_json(args)

	#Init renderer values (convert to Tensor, send to GPU)
	Renderer.init_renderer(args, device)

	# Create models
	model_A = FullyConnected(
		num_in		= 2 + 3 + 1, 	# normals/anisoMat/height
		num_out		= 2 + 3,		# normals/anisoMat
		nb_hidden_layers	= args.hid_nb_A,
		hidden_layer_size	= args.hid_size_A,
		non_lin				= args.non_linearity,
		is_B_net			= False
		).float().to(device)

	model_B = FullyConnected(
		num_in		= 2 + 3 + 1, 	# normals/anisoMat/height
		num_out		= 2 + 3,		# normals/anisoMat
		nb_hidden_layers 	= args.hid_nb_B,
		hidden_layer_size 	= args.hid_size_B,
		non_lin 			= args.non_linearity,
		is_B_net			= True
	).float().to(device)

	#Create output dir name
	args = file_utils.handle_empty_outdir(args)
	
	if args.mode == "eval":
		testing.eval_model_output_mips(args, model_A, model_B, device, isTrain=False)
		return

	##############
	## Training ##
	##############

	print("loading dataset")

	# create train dataset
	train_dataset = MIPNetDataset(args.data_path, args.train_file, args, device)

	# use Pytorch dataloader to help creating batches and randomize
	data_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size,
		shuffle=args.shuffle, pin_memory=False)

	all_params = [param for param in model_A.parameters()]
	all_params += [param for param in model_B.parameters()]

	#Define the optimizer and loss function
	optimizer = torch.optim.Adam(all_params, lr=args.learning_rate)

	# create validation dataset
	if args.val_file == args.train_file:
		val_dataset = train_dataset
	elif args.val_file:
		print("loading validation dataset")
		val_dataset = MIPNetDataset(
			args.data_path, args.val_file, args, device)
	else:
		val_dataset = None
		
	if val_dataset is not None:
		dataset_is_not_empty = val_dataset.get_nb_of_materials() > 0
		if dataset_is_not_empty and val_dataset is not train_dataset:
			val_data_loader = torch.utils.data.DataLoader(
				val_dataset, batch_size=args.batch_size,
				shuffle=args.shuffle, pin_memory=False)
		else:
			val_data_loader = None
	else:
		val_data_loader = None

	print("### output dir:", args.output_dir, "###")

	#Create the output dir if it doesn't exist
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	file_utils.save_training_metadata(args, model_A, model_B, train_dataset, val_dataset)

	training.train(model_A, model_B, data_loader, train_dataset, 
		val_data_loader, optimizer, args, device)
		
	return


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--mode", 				default="train",			type=str, choices=["train", "eval"])
	parser.add_argument("--data-path", 			default="./data/", 			type=str, help="where to look for data")
	parser.add_argument("--train-file", 		default="",					type=str, help="where to look for training data")
	parser.add_argument("--val-file", 			default="",					type=str, help="where to look for validation data")
	parser.add_argument("--eval-file", 			default="",					type=str, help="where to look for evaluation data")
	parser.add_argument("--model-path",  		default="./results/none/", 	type=str, help="path to networks epoch_X_mipnet_Y.model")
	parser.add_argument("--model-index", 		default="0",				type=str, help="provide index X networks epoch_X_mipnet_Y.model")
	parser.add_argument("--output-dir",  	   	default="./results/",		type=str, help="where to put output files")
	parser.add_argument("--manual-seed", 		default=1996,				type=int)
	parser.add_argument("--train-out-id", 		default=0,					type=int, help="material id for outputting visuals during training")
	parser.add_argument("--batch-size", 		default=16,					type=int, choices=[1, 4, 16, 64, 256, 1024, 4096])
	parser.add_argument("--hid-size-A", 		default=512,				type=int, help="size of the hidden layers in the A")
	parser.add_argument("--hid-nb-A",			default=2,					type=int, help="number of hidden layers in the A")
	parser.add_argument("--hid-size-B", 		default=1024,				type=int, help="size of the hidden layers in the B")
	parser.add_argument("--hid-nb-B",			default=3,					type=int, help="number of hidden layers in the B")
	parser.add_argument("--nb-levels", 			default=5, 					type=int, choices=[1,2,3,4,5,6,7,8], help='define the number of levels to train on, i.e., the number of terms in the loss')
	parser.add_argument("--num-epochs", 		default=50,					type=int, help="number of epochs for the training")
	parser.add_argument("--nb-wi-train", 		default=32,					type=int, choices=[8,16,32,64,128,256,512])
	parser.add_argument("--nb-wo-train", 		default=32,					type=int, choices=[8,16,32,64,128,256,512])
	parser.add_argument("--learning-rate", 		default=1e-4,				type=float)
	parser.add_argument("--disp-val", 			default=0.01,				type=float)
	parser.add_argument("--train-out-period", 	default=5,					type=int, help="output file during training per X periods")
	parser.add_argument("--render-res-thresh",	default=2,					type=int, choices=[4, 8, 16], help="do not render images smaller than this threshold")
	parser.add_argument("--map-size",			default=1024,				type=int, choices=[128, 256, 512, 1024, 2048, 4096], help="size (h/w) of input maps")
	parser.add_argument("--tf-period",			default=10,					type=int, help="register value using tensorboard each tf-period")
	parser.add_argument("--train-light-sampling",default="Hammersley",		type=str, choices=["Hammersley", "Fibonacci"],help='Type of point light direction sampling over the hemisphere')
	parser.add_argument("--loss-type", 			default="L1Loss",			type=str, choices=["L1Loss", "L2Loss", "FLIPLoss"],help='Type of loss used at the end of the pipeline')
	parser.add_argument("--non-linearity", 		default="LeakyReLU",		type=str, choices=["ReLU", "Sigmoid", "Hardsigmoid", "LeakyReLU"],help='Type of non-linearity used in the MLP')
	parser.add_argument("--renderer-type", 		default="GGX",				type=str, choices=["GGX", "Beckmann", "AshikhminShirley"], help='Type of renderer to use for training')
	parser.add_argument("--no-shuffle", 		dest='shuffle', 			action='store_false', help='shuffle dataset during training')
	parser.add_argument("--single-wo-training", dest='multi_wo', 			action='store_false', help='train using only the z vector as w_o direction')
	parser.add_argument("--neumip-tonemapper",	dest='neumip_tonemap', 		action='store_true', help='use Reinhard tonemapping and gamma correction in the loss')
	parser.add_argument("--colored-renders",	dest='colored_render', 		action='store_true', help='use colored renders during training')
	parser.add_argument("--cosine-weighted",	dest='cosine_weighted', 	action='store_true', help='use cosined weighted distribution of lights for training')
	parser.add_argument("--train-opengl-normals",dest='opengl_normals', 	action='store_true', help='input openGL normal maps instead of DirectX for training')
	parser.add_argument('--no-prog-bar', 		dest='no_prog_bar', 		action='store_true', help='no progress bar during training')
	parser.add_argument('--output-pdf',			dest='output_pdf', 			action='store_true', help='output graphs in pdf format')
	parser.add_argument('--output-train-maps',	dest='output_train_maps', 	action='store_true', help='output maps each train-out-period during training')
	parser.add_argument('--output-train-render',dest='output_train_render', action='store_true', help='output render each train-out-period during training')
	parser.add_argument('--eval-after-training',dest='eval_after_training', action='store_true', help='output mipmap images using the MLP')
	parser.add_argument('--eval-compute-flip',	dest='eval_compute_flip', 	action='store_true', help='output flip maps when using eval')
	parser.add_argument('--output-maps-pyramid',dest='output_maps_pyr', 	action='store_true', help='output avg and mlp mipmap svbrdf textures')
	parser.add_argument('--restart-from-file',	dest='restart', 			action='store_true', help='restart training from model checkpoint')
	parser.add_argument("--verbose", 			dest='verbose', 			action='store_true', help='output more text in console')
	parser.add_argument("--tensorboard-support",dest='use_tensorboard', 	action='store_true', help='output tensorboard loss statistics')
	parser.add_argument("--training-timestamp", dest='timestamp', 			action='store_true')

	## Training options
	parser.set_defaults(shuffle 			= True)
	parser.set_defaults(multi_wo			= True)

	## Loss options
	parser.set_defaults(neumip_tonemap		= False)
	parser.set_defaults(colored_render		= False)
	parser.set_defaults(cosine_weighted		= False)
	parser.set_defaults(opengl_normals		= False)

	## training common output option
	parser.set_defaults(eval_after_training	= False)
	parser.set_defaults(eval_compute_flip	= False)
	parser.set_defaults(output_maps_pyr		= False)

	## training detailed output option
	parser.set_defaults(no_prog_bar 		= False)
	parser.set_defaults(output_pdf 			= False)
	parser.set_defaults(output_train_maps	= False)
	parser.set_defaults(output_train_render	= False)

	## advanced training options
	parser.set_defaults(restart				= False)
	parser.set_defaults(verbose				= False)
	parser.set_defaults(use_tensorboard		= False)
	parser.set_defaults(timestamp			= False)

	args = parser.parse_args()

	if args.opengl_normals:
		print("###########################")
		print("## Using OpenGL normals  ##")
	else:
		print("###########################")
		print("## Using DirectX normals ##")

	if args.restart and args.mode != "train":
		file_utils.exit_with_message(
			"cannot restart training from checkpoint if mode!=train")

	if not args.data_path.endswith('/'):
		file_utils.exit_with_message(
			"Please add ending slash '/' to the data_path")

	# verify train/validation dir
	if args.mode == "train":
		if not os.path.isfile(args.train_file):
			file_utils.exit_with_message("Training file does not exists.")
		
	# verify eval and test conditions
	if args.mode == "eval":
		if not args.model_path:
			file_utils.exit_with_message(
				"Please provide model_path")
		if not args.model_index:
			file_utils.exit_with_message(
				"Please provide model_index")
		if args.model_path and not args.model_path.endswith('/'):
			file_utils.exit_with_message(
				"Please add ending slash '/' to the model_path")

	# verify output dir
	if args.output_dir and not args.output_dir.endswith('/'):
		file_utils.exit_with_message(
			"Please add ending slash '/' to the output_dir")

	main(args)