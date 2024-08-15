import warp as wp
import numpy as np

@wp.func
def relu(x: wp.float32) -> wp.float32:
    return wp.max(0.0, x)

@wp.kernel
def mlp_kernel(
    weights: wp.array2d(dtype=float),
    biases: wp.array(dtype=float),
    input: wp.array2d(dtype=float),
    output: wp.array2d(dtype=float)
):
    wp.mlp(weights, biases, relu, wp.tid(), input, output)

class MLP:

    # initializer
    def __init__(self, input_dim: int, hidden_dim: int, n_hidden_layers: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.output_dim = output_dim
    
    def get_weight_matrix_xavier(self, shape):
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def get_weight_matrices_xavier(self):
        mats_np = []
        
        # input layer
        mats_np.append(self.get_weight_matrix_xavier((self.input_dim, self.hidden_dim)))

        # hidden layers
        for _ in range(self.n_hidden_layers - 1):
            mats_np.append(self.get_weight_matrix_xavier((self.hidden_dim, self.hidden_dim)))
        
        # output layer
        mats_np.append(self.get_weight_matrix_xavier((self.hidden_dim, self.output_dim)))

        return mats_np
    
    def get_bias_vectors(self):
        return [np.zeros((self.hidden_dim,)) for _ in range(self.n_hidden_layers)] + [np.zeros((self.output_dim,))]

    def init_params(self):
        self.weights = [wp.array(mat, dtype=float) for mat in self.get_weight_matrices_xavier()]
        self.biases = [wp.array(vec, dtype=float) for vec in self.get_bias_vectors()]
    
    def set_params(self, weights: wp.array(dtype=float), biases: wp.array(dtype=float)):
        self.weights = weights
        self.biases = biases

    def n_weights(self) -> int:
        return self.hidden_dim * (self.input_dim + (self.n_hidden_layers - 1) * self.hidden_dim + self.output_dim)

    def n_biases(self):
        return self.hidden_dim * self.n_hidden_layers + self.output_dim
    
    def forward(self, input: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
        intermediate = wp.zeros((input.shape[0], self.hidden_dim), dtype=wp.float32)
        wp.launch(mlp_kernel, dim=input.shape[0], inputs=[self.weights[0], self.biases[0], input, intermediate], device="cuda")

        for i in range(self.n_hidden_layers):
            wp.launch(mlp_kernel, dim=input.shape[0], inputs=[self.weights[i], self.biases[i], intermediate, intermediate], device="cuda")
        
        wp.launch(mlp_kernel, dim=input.shape[0], inputs=[self.weights[-1], self.biases[-1], intermediate, output], device="cuda")
