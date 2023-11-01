## Problems with Fine-tuning Language Models:

Training a full network with billions of parameters, as seen in large language models, is computationally expensive. Additionally, storage requirements for checkpoints are costly, and the optimizer state requires additional memory to save momentum terms.

When dealing with multiple trained base models, switching between them necessitates unloading and reloading their entire weights. This process is time-consuming and resource-intensive.

## Minimum GPU Memory Requirements for Full Fine-Tuning:

### Weight:
- When employing mixed-precision training, the weight representation requires just 2 bytes. This reduces memory consumption compared to using 32-bit (single-precision) numbers.

### Weight Gradient:
- Similarly, weight gradients, crucial for optimization, are stored using 2-byte precision during mixed-precision training.

### Optimizer State with Adam:
- The Adam optimizer introduces further memory requirements. It demands 4 bytes for the original FP32 weight and 8 bytes for the first and second moment estimates.

Combining all these memory costs, you need 16 bytes of GPU memory per parameter. For a massive 15.5-billion-parameter model, this translates to a whopping 248GB of GPU memory. And remember, this estimate doesn't even consider the memory required for storing intermediate activations. Therefore, to fine-tune such a model, you'd require a minimum of 4 NVIDIA A100 80GB GPUs.

## QLORA (Quantized Low-Rank Adapters):




Qlora utilizes 4-bit quantization to compress a pretrained language model. The key idea is to freeze the language model parameters and add a relatively small number of trainable parameters in the form of Low-Rank Adapters.

### Gradient Precision:

QLoRA has one storage data type (usually 4-bit NormalFloat) for the base model weights and a computation data type (16-bit BrainFloat) used to perform computations. QLoRA dequantizes weights from the storage data type to the computation data type to perform the forward and backward passes, but only computes weight gradients for the LoRA parameters which use 16-bit bfloat. The weights are decompressed only when they are needed, therefore the memory usage stays low during training and inference.


During backpropagation, gradients are computed with respect to the model's parameters. Gradients can be both small and large values, so ensuring their precision is crucial for accurate model updates. In Qlora, the use of 16-bit BrainFloat ensures that gradients can be represented with sufficient precision.

### Adam vs SGD:

Adam optimizer, commonly used for training neural networks, introduces 2 additional values for each model parameter, and these are stored in 16-bit floats. This is a consideration for memory usage.

You might wonder why not use SGD (Stochastic Gradient Descent) for memory saving. For SGD optimizers, it's especially important to introduce a learning rate scheduler, like a cosine annealing schedule, which lowers the learning rate after each batch update. The decision to switch from AdamW to SGD depends on the number of trainable parameters, and for models like LoRA with low-rank values, the memory savings from this switch can be minimal compared to the benefits seen in pretraining large models.

## GPU Memory Requirements for Qlora (0.70% Trainable Parameters):

- Base model Weight: 0.5 bytes * 15.51B frozen params: 7.755 GB
- Adapter weight: 2 bytes * 0.11B trainable params: 0.22 GB
- Weight gradient: 2 bytes * 0.11B trainable params: 0.12 GB
- Optimizer state when using Adam: 4 bytes * 0.11B trainable params * 3: 1.32 GB

Adding all these up, you'd need around 10GB of GPU memory, suitable for a single A100 40GB GPU. Keep in mind that Qlora benefits from using A100 GPUs due to their compatibility with Flash Attention 2.

The reason for A100 40GB GPU being that the intermediate activations for long sequence lengths of 2048 and batch size of 4 for training lead to higher memory requirements.


For Qlora, combined with Flash Attention V2 and gradient checkpointing, a single A100 40GB GPU can accommodate the model, requiring a total of 26GB for a batch size of 4.

For full fine-tuning using FSDP (Fully Sharded Data Parallelism) with Flash Attention V2 and Gradient Checkpointing, the memory required per GPU ranges between 70GB to 77.6GB, with a per-gpu_batch_size of 1.


## LORA

![Lora-img](https://github.com/prasad4fun/ML-Portfolio/assets/12726341/7fb90a01-88fc-4696-b263-597839ea02f0)


## Advantages of Qlora:

- Reduced number of parameters to train and store.
- Introduction of low-rank values results in more linearly independent vectors, contributing to memory savings.
- Smaller storage requirements and faster backpropagation.
- Switchable adapters for improved model flexibility.

## Key Takeaways:

- LoRA was initially enabled for Key and Query matrices in multi-head self-attention blocks.
- Baselines can be further improved by enabling LORA for the Value matrix, the projection layers, and the linear layers. Experiment with different settings to see if the model's performance improves.

### Alpha (α):

- Adjusting the hyperparameter "alpha" allows you to strike a balance between fitting the data and preventing overfitting by regularizing the model.
- As a rule of thumb, choosing an alpha approximately twice as large as the rank is common when fine-tuning LLMs. Experiment with various combinations of rank (r) and alpha to find the settings that yield the best performance.

## Implementing LoRA with PyTorch
Below is a PyTorch code tutorial on how to implement LoRA in your model.

```python
import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
        super().__init__()
        
        # Initialize the LoRA parameters A and B
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        
        # Set the scaling factor based on alpha and rank
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return the modified weights: W + (B * A) * scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights


# Register LoRA parameterization for a specific layer
def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1):
    # Only add the parameterization to the weight matrix, ignore the Bias
    features_in, features_out = layer.weight.shape
    return LoRA(features_in, features_out, rank=rank, alpha=lora_alpha, device=device)

# Instantiate your model
your_model = YourModel()

# Register LoRA parameterization for a specific layer (e.g., linear1)
parametrize.register_parametrization(
    your_model.linear1, "weight", linear_layer_parameterization(your_model.linear1, device)
)
```

In this code, we define the `LoRA` module, which takes care of modifying the layer's weights. We then create a sample model, `YourModel`, and register the LoRA parameterization for a specific layer, in this case, `linear1`. The code allows you to easily incorporate LoRA into your PyTorch models, reducing the number of parameters and saving memory during training. Adjust the input and output dimensions to match your specific use case.
