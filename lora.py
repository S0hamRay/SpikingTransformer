import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Wraps a linear layer with a low-rank adapter.
    y = xW^T + b + (alpha/r) * x A^T B^T
    W is frozen; only A and B are trainable.
    """
    def __init__(self, in_features, out_features, r=4, alpha=8, bias=True):
        super().__init__()
        self.r = r
        self.scale = alpha / r

        # Original (frozen) linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

        # Low-rank decomposition: delta_W = B @ A, shape (out, in)
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))

        # Init: A ~ Kaiming, B = 0 so LoRA starts as a no-op
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + self.scale * lora_out


def load_pretrained_weights(lora_layer, pretrained_linear):
    """Copy frozen weights from a pretrained nn.Linear into the LoRA layer."""
    lora_layer.linear.weight.data.copy_(pretrained_linear.weight.data)
    if pretrained_linear.bias is not None:
        lora_layer.linear.bias.data.copy_(pretrained_linear.bias.data)


if __name__ == "__main__":
    torch.manual_seed(0)

    in_f, out_f, r = 16, 8, 4
    pretrained = nn.Linear(in_f, out_f)
    lora_layer = LoRALinear(in_f, out_f, r=r, alpha=8)
    load_pretrained_weights(lora_layer, pretrained)

    x = torch.randn(2, in_f)

    # At init, B=0, so LoRA output should exactly match the frozen base layer
    with torch.no_grad():
        assert torch.allclose(lora_layer(x), pretrained(x))
    print("Sanity check passed: LoRA is a no-op at init.")

    # Only A and B require grad
    trainable = [n for n, p in lora_layer.named_parameters() if p.requires_grad]
    print("Trainable params:", trainable)

    # A quick training step to show gradients flow only through A, B
    target = torch.randn(2, out_f)
    opt = torch.optim.SGD([lora_layer.A, lora_layer.B], lr=0.1)
    loss = nn.functional.mse_loss(lora_layer(x), target)
    loss.backward()
    opt.step()
    print("Loss after one step:", loss.item())
