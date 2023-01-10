import torch

class FullyConnected(torch.nn.Module):

	non_linearity_dict = {
        "ReLU": 		torch.nn.ReLU(),
        "Sigmoid": 		torch.nn.Sigmoid(),
        "Hardsigmoid": 	torch.nn.Hardsigmoid(),
		"LeakyReLU": 	torch.nn.LeakyReLU()
    }

	def define_mlp(self):

		if self.nb_hidden_layers == 0:
			print("unsupported nb_hidden_layers=0")
			exit(0)
		
		self.init_conv_1x1 = torch.nn.Conv2d(
				self.num_in, self.hidden_layer_size, 
				kernel_size = 	(1, 1), 
				stride = 		(1, 1))
		
		self.init_conv_2x2 = torch.nn.Conv2d(
			self.num_in, self.hidden_layer_size, 
			kernel_size = 	(2, 2), 
			stride = 		(2, 2))

		if self.is_B_net:
			self.init_conv_4x4 = torch.nn.Conv2d(
				self.num_in, self.hidden_layer_size, 
				kernel_size = 	(4, 4), 
				stride = 		(4, 4))
		
		if self.is_B_net:
			self.conv_data = [] # use forward_B()
		else:
			self.conv_data = [self.init_conv_2x2]

		seq = self.conv_data + [self.non_linearity]

		for _ in range(0, self.nb_hidden_layers - 1):
			seq.append(torch.nn.Conv2d(
				self.hidden_layer_size, self.hidden_layer_size, 1))
			seq.append(self.non_linearity)

		if self.nb_hidden_layers >= 1:
			seq.append(torch.nn.Conv2d(self.hidden_layer_size, self.num_out, 1))
		
		self.mlp = torch.nn.Sequential(*seq)


	def __init__(self, num_in, num_out, nb_hidden_layers, hidden_layer_size, non_lin, is_B_net):

		super(FullyConnected, self).__init__()

		self.num_in 			= num_in
		self.num_out 			= num_out
		self.nb_hidden_layers 	= nb_hidden_layers
		self.hidden_layer_size 	= hidden_layer_size
		self.is_B_net 			= is_B_net
		self.non_linearity 		= self.non_linearity_dict[non_lin]
		
		self.define_mlp()


	def forward(self, x, y = None, z = None):
		if self.is_B_net:
			return self.forward_B(x, y, z)
		else:
			return self.mlp(x)

	def forward_B(self, lod0, lod1, lod2):
		# 4x4 -> 1x1
		res1 = self.init_conv_4x4(lod0)
		# 2x2 -> 1x1
		res2 = self.init_conv_2x2(lod1)
		# 1x1 -> 1x1
		res3 = self.init_conv_1x1(lod2)
		return self.mlp(res1 + res2 + res3)