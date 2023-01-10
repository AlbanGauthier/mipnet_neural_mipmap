import torch
import time
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from . import file_utils
from . import light_utils
from . import data_utils
from . import testing
from . import renderer

from mipnet.losses import CustomLoss

def train_one_epoch(
	model_1, model_2, loss_func, optimizer, 
	dataset, data_loader, wi_train, val_data_loader, 
	wo_train, args, device, epoch_nb, trainwriter, valwriter):

	model_1.train()
	model_2.train()

	if not args.no_prog_bar:
		prog_bar = tqdm(total=len(data_loader), leave=False)

	it_train = 0
	it_val 	 = 0
	total_train_loss = 0
	total_val_loss 	 = 0
	current_mean_train_loss = 0
	current_mean_val_loss 	= 0

	nb_levels = int(np.log2(dataset.get_tile_width()))
	nb_train_samples = len(data_loader)

	use_validation_set = val_data_loader is not None

	if use_validation_set:
		nb_val_samples = len(val_data_loader)
		val_dataloader_iter = iter(val_data_loader)
	else:
		nb_val_samples = -1

	for L0_batch in data_loader:

		## Clear all accumulated gradients
		optimizer.zero_grad()
		
		L0_batch = L0_batch.to(device)

		# mlp_output: (N, dims_to_learn, half_tile_width, half_tile_width)
		mlp_Li_delta_list, mlp_Li_bar_list = data_utils.process_in_mlp_multi_loss(
			L0_batch, model_1, model_2, nb_levels)

		if args.colored_render:
			light_col = data_utils.generate_rand_light_col().to(device)
		else:
			light_col = renderer.Renderer.white_light
			
		light_intensity = 9 * torch.rand(1).item() + 1

		L1_bar = data_utils.average_patches(L0_batch, ks=2, dim1=1, dim2=2)
		Li_bar = L1_bar

		for i in range(len(mlp_Li_bar_list)):
			Li_bar = data_utils.average_patches(Li_bar, ks=2, dim1=1, dim2=2)
			mlp_Li_bar_list[i] = data_utils.add_missing_channels_to_mlp_out(mlp_Li_bar_list[i], Li_bar)

		loss = loss_func(
			mlp_Li_delta_list, L0_batch, L1_bar, mlp_Li_bar_list,
			wi_train, wo_train, light_col, light_intensity, args)

		# Backpropagate the loss
		loss.backward()

		# Adjust parameters according to the computed gradients
		optimizer.step()

		if args.use_tensorboard:
			current_mean_train_loss += loss.item()
			if it_train % args.tf_period==0 and it_train!=0:
				trainwriter.add_scalar(
					"", current_mean_train_loss / args.tf_period, 
					(epoch_nb-1)*nb_train_samples+it_train)
				current_mean_train_loss = 0

		total_train_loss += loss.item()

		if use_validation_set:
			if it_train / nb_train_samples >= it_val / nb_val_samples:
				L0_val_batch = next(val_dataloader_iter)
				L0_val_batch = L0_val_batch.to(device)
				model_1.eval()
				model_2.eval()
				with torch.no_grad():
					mlp_Li_delta_list, mlp_Li_bar_list = data_utils.process_in_mlp_multi_loss(
						L0_val_batch, model_1, model_2, nb_levels)
					L1_bar = data_utils.average_patches(
						L0_val_batch, ks=2, dim1=1, dim2=2)
					Li_bar = L1_bar
					for i in range(len(mlp_Li_bar_list)):
						Li_bar = data_utils.average_patches(
							Li_bar, ks=2, dim1=1, dim2=2)
						mlp_Li_bar_list[i] = data_utils.add_missing_channels_to_mlp_out(mlp_Li_bar_list[i], Li_bar)
					val_loss = loss_func(mlp_Li_delta_list, L0_val_batch, 
						L1_bar, mlp_Li_bar_list, wi_train, wo_train,
						renderer.Renderer.white_light, light_intensity, args)
					if args.use_tensorboard:
						current_mean_val_loss += val_loss.item()
						if it_train % args.tf_period==0 and it_train!=0:
							valwriter.add_scalar(
								"", current_mean_val_loss / args.tf_period, 
								(epoch_nb-1)*nb_train_samples+it_train)
							current_mean_val_loss = 0
					total_val_loss += val_loss.item()
				it_val += 1
				model_1.train()
				model_2.train()

		it_train += 1

		if not args.no_prog_bar:
			prog_bar.update(1)

	if not args.no_prog_bar:
		prog_bar.close()

	return total_train_loss / nb_train_samples, total_val_loss / nb_val_samples


def train(model_1, model_2, data_loader, train_dataset, 
	val_data_loader, optimizer, args, device):

	nb_mats = train_dataset.get_nb_of_materials()
	if args.train_out_id > nb_mats:
		file_utils.exit_with_message("wrong train_out_id")

	start = 0
	if args.restart:
		print("#########################")
		print("restarting training from epoch: " + args.model_index)
		print("#########################")
		model_1.load_state_dict(torch.load(
            args.model_path + "epoch_" + \
            args.model_index + "_mipnet_A.model"))
		model_2.load_state_dict(torch.load(
			args.model_path + "epoch_" + \
			args.model_index + "_mipnet_B.model"))
		start = int(args.model_index)

	# saving references render and averaged maps
	with torch.no_grad():
		data_utils.save_references(
			args, train_dataset, 
			level_str="1")
	# 	data_utils.output_avg_maps(
	# 		args, train_dataset, 
	# 		renderer.Renderer.white_light,
	# 		kernel_size = 2,
	# 		level_str="1")
	
	wi_train = 1.5 * light_utils.generate_hemisphere_pts(
		args.train_light_sampling, 
		args.nb_wi_train, near_z=True,
		cosine_weighted=args.cosine_weighted).float().to(device)

	if args.multi_wo:
		wo_train = light_utils.generate_hemisphere_pts(
			"Hammersley", args.nb_wo_train,
			near_z=True, cosine_weighted=True).float().to(device)
	else:
		wo_train = renderer.Renderer.z_vector

	train_img_output = args.output_train_maps or args.output_train_render

	use_validation_set = val_data_loader is not None

	trainwriter = None
	valwriter = None

	if args.use_tensorboard:
		trainwriter = SummaryWriter(args.output_dir + "train")
		if use_validation_set:
			valwriter = SummaryWriter(args.output_dir + "val")
			
	print("Start training")
	start_time = time.time()
	last_time = start_time

	loss_func = CustomLoss(device, args).loss_multiscale

	for epoch in range(start + 1, args.num_epochs + start + 1):

		loss_list = train_one_epoch(
			model_1, model_2, loss_func, optimizer, train_dataset, 
			data_loader, wi_train, val_data_loader, wo_train, 
			args, device, epoch, trainwriter, valwriter)

		file_utils.update_loss_json(epoch, loss_list, args.output_dir)

		current_time = time.time() - last_time
		current_time_str = str(
			datetime.timedelta(seconds=int(current_time)))
		
		out_str = "Epoch:" + str(epoch) + "/" 
		out_str += str(args.num_epochs + start)
		out_str += " time {}".format(current_time_str)
		out_str += " Train L1 Loss: "
		out_str += "{:.4f}".format(loss_list[0])
		
		if use_validation_set:
			out_str += " Validation L1 Loss: "
			out_str += "{:.4f}".format(loss_list[1])
		
		print(out_str)
		
		if epoch % args.train_out_period == 0:
			file_utils.save_models(model_1, model_2, epoch, args.output_dir)

		if ((epoch % args.train_out_period) == 0) and train_img_output:
			with torch.no_grad():
				testing.eval_output_maps_and_render(
					model_1, model_2, train_dataset, args, str(epoch))

		last_time = time.time()

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	print('Total training time {}'.format(total_time_str))

	if args.use_tensorboard:
		trainwriter.close()
		if use_validation_set:
			valwriter.close()

	if not args.restart:
		file_utils.update_json_with_data(
			{"total time": total_time_str}, args.output_dir)

	file_utils.save_models(model_1, model_2, args.num_epochs + start, args.output_dir)
	
	if train_img_output:
		with torch.no_grad():
			testing.eval_output_maps_and_render(
				model_1, model_2, train_dataset, args, 
				str(args.num_epochs + start))

	del data_loader
	del train_dataset

	file_utils.output_loss_graph(args, args.output_dir, hasValidationData=use_validation_set)

	if args.eval_after_training:
		testing.eval_model_output_mips(args, model_1, model_2, device, isTrain=True)

	return