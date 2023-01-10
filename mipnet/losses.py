import torch

from flip_pytorch import flip_loss

from mipnet import data_utils
from mipnet import render_utils


class CustomLoss():

	loss_dict = {
		"L1Loss": torch.nn.L1Loss(),
		"L2Loss": torch.nn.MSELoss(),
		"FLIPLoss": flip_loss.LDRFLIPLoss()
	}


	def __init__(self, device, args):
		self.sumLoss = self.loss_dict[args.loss_type].to(device)
		self.l1_loss = self.loss_dict["L1Loss"].to(device)
		self.flip_loss = self.loss_dict["FLIPLoss"].to(device)
		self.loss_str = args.loss_type
		self.height_idx = 5


	def loss_multiscale(self, mlp_Li_delta_list, L0, L1_bar, mlp_Li_bar_list,
		wi_array, wo_array, light_col, light_intensity, args):

		L0 			= data_utils.fold_tensor(L0)
		L1_bar 		= data_utils.fold_tensor(L1_bar)

		for i in range(len(mlp_Li_delta_list)):
			mlp_Li_delta_list[i] = data_utils.fold_tensor(mlp_Li_delta_list[i])
		for i in range(len(mlp_Li_bar_list)):
			mlp_Li_bar_list[i] = data_utils.fold_tensor(mlp_Li_bar_list[i])

		render_1 = data_utils.tonemapped_render_from_mlp_output(
			mlp_Li_delta_list[0], L1_bar, wi_array, wo_array, 
			light_col, args, light_intensity)

		# compute L1 Groundtruth
		gt_render_1 = data_utils.compute_L1_gt_render(
			L0, wi_array, wo_array, light_col, args, 
			light_intensity=light_intensity)

		gt_render_i = gt_render_1
		
		if args.neumip_tonemap:
			gt_render_1_tn = render_utils.neuMIPTonemapper(gt_render_1)
		else:
			gt_render_1_tn = render_utils.process_raw_render(gt_render_1)

		if self.loss_str == "FLIPLoss":
			render_1 = render_1.permute(3, 2, 0, 1)
			gt_render_1_tn = gt_render_1_tn.permute(3, 2, 0, 1)
		else:
			render_1 = render_1.view(-1)
			gt_render_1_tn = gt_render_1_tn.view(-1)

		loss_val = self.sumLoss(render_1, gt_render_1_tn)

		for i in range(len(mlp_Li_bar_list)):

			render_i = data_utils.tonemapped_render_from_mlp_output(
				mlp_Li_delta_list[i+1], mlp_Li_bar_list[i], wi_array, wo_array,
				light_col, args, light_intensity)

			# compute Li Groundtruth
			gt_render_i = data_utils.average_patches(gt_render_i)

			if args.neumip_tonemap:
				gt_render_i_tn = render_utils.neuMIPTonemapper(gt_render_i)
			else:
				gt_render_i_tn = render_utils.process_raw_render(gt_render_i)

			if self.loss_str == "FLIPLoss":
				render_i = render_i.permute(3, 2, 0, 1)
				gt_render_i_tn = gt_render_i_tn.permute(3, 2, 0, 1)
			else:
				render_i = render_i.view(-1)
				gt_render_i_tn = gt_render_i_tn.view(-1)

			loss_val += self.sumLoss(render_i, gt_render_i_tn)

		assert not torch.any(torch.isfinite(loss_val) == False)
		assert not torch.isnan(loss_val)
		
		return loss_val / len(mlp_Li_delta_list)


	def fullres_diff(self, mlp_out_Li, Li_bar, L0, args, 
		wi_array, wo_array, light_col, gt_target_lod):

		# compute MLP out render
		render_Li = data_utils.tonemapped_render_from_mlp_output(
			mlp_out_Li, Li_bar, 
			wi_array, light_col, args)

		# compute Groundtruth
		gt_render = data_utils.compute_full_lod_gt_render(
			gt_target_lod, L0, wi_array, wo_array, light_col, args)

		if args.neumip_tonemap:
			gt_render = render_utils.neuMIPTonemapper(gt_render)
		else:
			gt_render = render_utils.process_raw_render(gt_render)

		flip_Li = render_Li.permute(3, 2, 0, 1)
		flip_gt = gt_render.permute(3, 2, 0, 1)
		flip_loss = self.flip_loss(flip_Li, flip_gt)
		
		render_Li = render_Li.view(-1)
		gt_render = gt_render.view(-1)
		l1_loss = self.l1_loss(render_Li, gt_render)

		assert not torch.any(torch.isfinite(l1_loss) == False)
		assert not torch.isnan(l1_loss)
		assert not torch.any(torch.isfinite(flip_loss) == False)
		assert not torch.isnan(flip_loss)

		return l1_loss.cpu().item(), flip_loss.cpu().item()
