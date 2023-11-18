import torchvision
from types import MethodType

from torchvision.models import VisionTransformer

import torch
from torch._C._profiler import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile

import fancy_unet


class Printer(torch.nn.Module):
    def __init__(self, layer=None, on_end=False):
        super().__init__()
        self.layer = layer
        self.on_end = on_end

    def forward(self, x):
        print(x.shape)
        if self.layer is not None:
            x = self.layer(x)
        if self.on_end:
            print(x.shape)
        return x


def get_base_model():
    return torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
        progress=True
    )


def insert_printers(model: VisionTransformer):
    model.encoder.layers = torch.nn.Sequential(
        *(Printer(layer) for layer in model.encoder.layers)
    )
    model = Printer(model, on_end=True)
    return model


def convert_to_segmentation_model(model: VisionTransformer, output_channels: int):
    class DummyClass(VisionTransformer):

        def add_reverse_conv_proj(self, output_channels: int):
            self.add_module(
                'rev_conv_proj',
                torch.nn.ConvTranspose2d(
                    self.hidden_dim,
                    output_channels,
                    kernel_size=self.patch_size,
                    stride=self.patch_size
                )
            )

        def process_output(self, shape, x: torch.Tensor):
            n, c, h, w = shape
            n_h = h // self.patch_size
            n_w = w // self.patch_size

            #  (n, hidden_dim, (n_h * n_w)) -> (n, hidden_dim, n_h, n_w)
            x = x.reshape(n, self.hidden_dim, n_h, n_w)
            #  (n, hidden_dim, n_h, n_w) -> (n, c, h, w)
            x = self.rev_conv_proj(x)

            return x

        def forward(self, x: torch.Tensor):
            shape = x.shape

            # Reshape and permute the input tensor
            x = self._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.encoder(x)

            x = x[:, 1:]

            x = self.process_output(shape, x)

            return x

    model.add_reverse_conv_proj = MethodType(DummyClass.add_reverse_conv_proj, model)
    model.process_output = MethodType(DummyClass.process_output, model)
    model.forward = MethodType(DummyClass.forward, model)

    model.add_reverse_conv_proj(output_channels)

    return model


def main():
    with torch.device('cuda'):
        base_model = fancy_unet.Unet()

        # Test
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True
        ) as prof:
            with record_function("model_inference"):
                example = torch.zeros((4, 1, 384, 384))
                out = base_model(example)
                loss = torch.nn.functional.mse_loss(out, torch.empty_like(out))
                loss.backward()
                print(loss.item())

        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

        base_model = get_base_model()
        print(base_model)

        model = insert_printers(base_model)
        example = torch.zeros((4, 3, 384, 384))
        out = model(example)
        # in 3 384 384
        # conv_proj 768 24 24
        # in encoder 576+1 768
        # heads output 768

        out.sum().backward()

        model = convert_to_segmentation_model(base_model, 10)
        model = insert_printers(model)
        out = model(example)
