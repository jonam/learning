# Feed Forward

## Feed Forward basics

```
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

The code snippet we've defines a neural network module in PyTorch, utilizing the `nn.Module` class as a base. Let's break down both parts for clarity.

### The `self.net` Definition

This module uses `nn.Sequential` to create a simple feedforward neural network pipeline. `nn.Sequential` is a container that holds and organizes modules in a sequential manner, where the output of one module becomes the input to the next. This makes it easy to define a straightforward flow of data without manually coding the passing of data from one layer to the next. Here's what each part does:

1. **`nn.Linear(n_embd, 4 * n_embd)`**: This is a linear (or fully connected) layer that transforms the input tensor from a dimension of `n_embd` to `4 * n_embd`. If `n_embd` is the size of your input features, this layer increases the dimensionality of the input by a factor of 4. This is often done to allow the network to learn more complex representations.

2. **`nn.ReLU()`**: After the linear transformation, the ReLU (Rectified Linear Unit) activation function is applied. ReLU is defined as `f(x) = max(0, x)`, and it introduces non-linearity into the model, allowing it to learn more complex functions. It essentially replaces all negative values in the input tensor with zero, which helps with gradient flow during backpropagation and generally leads to faster convergence.

3. **`nn.Linear(4 * n_embd, n_embd)`**: This is another linear layer that projects the data back from the increased dimensionality (`4 * n_embd`) to the original dimensionality (`n_embd`). This kind of "expansion and contraction" is common in neural network designs, as it allows the network to mix and transform the features before reducing the dimensionality again, hopefully retaining useful information while doing so.

4. **`nn.Dropout(dropout)`**: Dropout is a regularization technique used to prevent overfitting. During training, this layer randomly sets a fraction of the input units to 0 with a probability of `dropout` at each step. This helps to make the model more robust and less dependent on any single element of the input. The `dropout` parameter specifies the probability of an element to be zeroed.

### The `forward` Method

The `forward` method is a special method in PyTorch `nn.Module` that defines how the input `x` flows through the network or, in this case, through the `nn.Sequential` container.

```python
def forward(self, x):
    return self.net(x)
```

When you pass an input tensor `x` to this module, the `forward` method is automatically called. Inside this method, `x` is passed through the `self.net` sequence of layers you defined earlier. The input `x` first goes through the initial linear layer, gets activated by ReLU, passes through the second linear layer, and finally goes through dropout, with the output of each step being the input to the next. The final output of `self.net(x)` is then returned as the output of the `forward` method.

This design pattern is fundamental in PyTorch for defining custom models, allowing for clear and concise definition of model architectures.

## How are the weights determined? That's a lot of weights. Are these random numbers again?

Yes, initially, the weights in a neural network layer, such as `nn.Linear(n_embd, 4 * n_embd)`, are typically initialized with random numbers, following specific initialization strategies. The choice of these strategies is crucial for the successful training of deep neural networks, as they can significantly influence the convergence rate and the overall performance of the model. Let's discuss how the weights are determined and the role of randomness in their initialization:

### Initial Randomness

1. **Random Initialization**: The weights are usually initialized randomly to break symmetry. If all weights were initialized to the same value, each neuron would learn the same features during training, which is not desirable. Random initialization ensures that neurons can start learning different aspects of the input data.

2. **Initialization Strategies**: There are several strategies for initializing weights, and they often depend on the activation function used in the network. Common strategies include:
   - **Xavier/Glorot Initialization**: This strategy is designed for networks with Sigmoid or Tanh activation functions. It initializes the weights to values chosen from a distribution with zero mean and a variance that depends on the number of input and output units, specifically `var(W) = 2 / (n_in + n_out)` where `n_in` and `n_out` are the number of input and output units, respectively.

The Xavier formula effectively averages the considerations for both the input and output dimensions of the layer, providing a balance that empirically works well for maintaining stable gradients in networks using Sigmoid or Tanh activations.
   - **He Initialization**: Tailored for networks with ReLU activation functions, it initializes the weights from a distribution with zero mean and variance `2 / n_in`, where `n_in` is the number of input units. This aims to address the vanishing gradient problem in deep networks with ReLU activations.

### Training and Adjustment

- **Learning Through Training**: The initial weights, although randomly assigned, are not fixed. They are updated during the training process through backpropagation. As the network is exposed to input data and corresponding feedback (errors), the weights are adjusted to minimize the loss function. This adjustment is typically performed using optimization algorithms like Stochastic Gradient Descent (SGD), Adam, etc.

- **Role of Backpropagation**: Through backpropagation, the network calculates the gradient of the loss function with respect to each weight, indicating how the loss would change if the weight were increased or decreased. The optimization algorithm then uses this information to update the weights in a direction that minimally reduces the loss.

### Large Number of Weights

Indeed, expanding the dimensionality increases the number of weights significantly, which can lead to a model with a large number of parameters. This expansion allows the network to learn complex patterns and relationships in the data but also requires more computational resources and data to train effectively. The increased capacity (number of parameters) can lead to more powerful models but also raises concerns about overfitting, which is usually mitigated through techniques like dropout, regularization, and careful model design and training strategies.

In summary, while the initial weight values are determined randomly based on specific initialization strategies, their final values are learned from the data through the process of training the network.

### Vanishing Gradient Problem

The vanishing gradient problem is a challenge that occurs in the training of artificial neural networks, especially in deep learning models. It refers to the situation where the gradients (the values used to update the weights) of the network's loss function decrease to zero as they are propagated back through the network during training. This issue is particularly prevalent in networks that use certain activation functions, like the sigmoid or tanh functions, for their neurons.

Here's why it happens:

1. **Gradient Propagation**: In backpropagation, gradients of the loss function with respect to the network's parameters are calculated and propagated back through the network from output to input. These gradients are used to update the weights in each layer.

2. **Activation Functions**: Activation functions like sigmoid or tanh have derivatives that can be very small. Sigmoid, for instance, squashes its input to a range between 0 and 1, and its gradient is maximum at 0.25. This means that when the input to the sigmoid is very large or very small, the gradient can become very close to 0.

3. **Multiplicative Effect**: During backpropagation, gradients from the output layer are multiplied successively by the gradients of each layer's weights to compute the update for the weights in the initial layers. If these gradients are small (<1), multiplying many of them together (as in deep networks) will make the gradient signal that reaches the earlier layers exponentially smaller. This is the vanishing gradient problem.

Consequences of the vanishing gradient problem include:

- **Slow Training**: If the gradients are vanishing, the weights of the early layers in the network are updated very slowly, if at all. This can make training deep networks very slow and sometimes impractical.
- **Poor Performance**: Because early layers learn more slowly, the network may not effectively learn the features in the input data that are critical for making accurate predictions or classifications. This can result in suboptimal performance of the model.

To mitigate the vanishing gradient problem, several strategies can be employed, such as:

- **Using ReLU Activation**: The Rectified Linear Unit (ReLU) and its variants (e.g., Leaky ReLU, ELU) have been found to help mitigate the vanishing gradient problem because they do not saturate in the positive domain. ReLU, for instance, has a gradient of 0 for negative inputs and 1 for positive inputs.
- **Careful Initialization**: Initializing the weights of the network in a way that prevents the gradients from becoming too small too quickly.
- **Batch Normalization**: Normalizing the inputs of each layer to have mean 0 and variance 1 to ensure that the gradients do not vanish or explode.
- **Skip Connections**: Architectures like ResNet introduce skip connections that allow gradients to flow directly through the network without passing through problematic activation functions, mitigating the vanishing gradient problem.

These approaches have enabled the successful training of very deep networks, which was challenging in the early days of deep learning.
