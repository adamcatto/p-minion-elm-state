import numpy as np


dot_product = np.dot


class Kernel:
	def __init__(self, kernel_array: np.array):
		self.kernel_array = kernel_array
		self.receptive_field_size = kernel_array.shape

	def __repr__(self):
		return self.kernel_array

	def __str__(self):
		return str(self.kernel_array)
	
	def convolve(self, local_image_patch):
		assert self.receptive_field_size == local_image_patch.shape
		convolution = dot_product(self.kernel_array, local_image_patch)
		return convolution


class ConvLayer:
	def __init__(self, dimensionality, inputs, num_filters, kernel_size, padding_size, padding_type, stride_length):
		self.dimensionality = dimensionality
		self.inputs = inputs
		self.input_size = inputs.shape
		self.num_filters = num_filters
		if isinstance(kernel_size, int):
			if dimensionality != 1:
				self.kernel_size = (kernel_size) * dimensionality
			else:
				self.kernel_size = kernel_size
		else:
			self.kernel_size = kernel_size
		self.padding_size = padding_size
		self.padding_type = padding_type
		self.stride_length = stride_length

		# need to make sure shapes match
		self.output_size = np.add(1, (self.input_size - self.kernel_size + 2 * stride_length) / 2)
		
		self.kernels = np.array([Kernel(np.zeros(kernel_size))])

	def forward(self):
		pass




def unpool_nearest_neighbor(matrix):
	height, width = matrix.shape
