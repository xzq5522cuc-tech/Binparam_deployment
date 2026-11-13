"""Inference-only PyMAF partitions for Bolt + native C++ SMPL/MAF deployment."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidWrapper(nn.Module):
    """Image -> the three 256-channel spatial feature maps (14, 28, 56)."""

    def __init__(self, pymaf):
        super().__init__()
        self.feature_extractor = pymaf.feature_extractor
        self.deconv0 = pymaf.deconv_layers[0:3]
        self.deconv1 = pymaf.deconv_layers[3:6]
        self.deconv2 = pymaf.deconv_layers[6:9]

    def forward(self, image):
        spatial, _ = self.feature_extractor(image)
        feature14 = self.deconv0(spatial)
        feature28 = self.deconv1(feature14)
        feature56 = self.deconv2(feature28)
        return feature14, feature28, feature56


class ParameterStageWrapper(nn.Module):
    """Raw point features + previous parameters -> updated parameters.

    This is the exact TEST_NUM=63 inference path with training-only entropy
    calculation removed. Its standardized feature tensor and parameter outputs
    are mathematically identical to Regressor.forward, but it never invokes
    rotation conversion, SMPL, projection, or joint regression.
    """

    def __init__(
        self, maf_extractor, regressor, stable_binary=False, sign_epsilon=1e-4,
        bolt_bnn_linear_as_conv=False,
    ):
        super().__init__()
        self.maf_extractor = maf_extractor
        self.regressor = regressor
        self.stable_binary = stable_binary
        self.sign_epsilon = float(sign_epsilon)
        self.bolt_bnn_linear_as_conv = bool(bolt_bnn_linear_as_conv)
        for name, layer in (("fc1_weight", regressor.fc1), ("fc2_weight", regressor.fc2)):
            if hasattr(layer, "weight_bin") and hasattr(layer, "output_scale"):
                weight_bin = layer.weight_bin.detach().clone()
                output_scale = layer.output_scale.detach().clone()
            else:
                real = layer.weight.detach()
                weight_bin = torch.sign(real)
                output_scale = real.abs().mean(dim=1)
            self.register_buffer(name + "_bin", weight_bin)
            self.register_buffer(name + "_scale", output_scale)
            bias = layer.bias.detach().clone() if layer.bias is not None else torch.zeros(
                layer.out_features, dtype=weight_bin.dtype, device=weight_bin.device
            )
            self.register_buffer(name + "_bias", bias)

        if self.bolt_bnn_linear_as_conv:
            self._register_bolt_conv_weight("fc1", self.fc1_weight_bin)
            self._register_bolt_conv_weight("fc2", self.fc2_weight_bin)

    def _register_bolt_conv_weight(self, name, weight):
        """Convert a binary Linear weight to a Bolt-compatible packed Conv.

        An earlier [N, C, 1, 1] rewrite was accepted and weight-packed by
        X2bolt, but failed at runtime because the Stage graph propagated NCHW
        into Bolt's AArch64 NCHWC8-only BNN kernel.  Reshaping the same dot
        product to 32 channels gives the backend a regular packable tensor
        without changing its value.  The spatial kernel is padded to 8k+1,
        which is the other Bolt BNN constraint.
        """
        out_features, in_features = weight.shape
        packed_channels = 32
        kernel_width = (in_features + packed_channels - 1) // packed_channels
        kernel_width = ((kernel_width - 1 + 7) // 8) * 8 + 1
        padded_in = packed_channels * kernel_width
        padding = padded_in - in_features
        if out_features % 16:
            raise ValueError(f"{name}: output size {out_features} is not divisible by 16")
        if padding % 2:
            raise ValueError(f"{name}: channel padding {padding} must be even")
        if padding:
            # Added activations are +1. Equal numbers of +1 and -1 padding
            # weights make their contribution exactly zero for every output.
            pad_weight = torch.ones(
                out_features, padding, dtype=weight.dtype, device=weight.device
            )
            pad_weight[:, padding // 2:] = -1
            weight = torch.cat((weight, pad_weight), dim=1)
        self.register_buffer(
            name + "_conv_weight",
            weight.reshape(out_features, packed_channels, 1, kernel_width),
        )
        setattr(self, name + "_padding_value", padding)
        setattr(self, name + "_kernel_width", kernel_width)

    @staticmethod
    def _binary_linear_as_conv(x, weight, scale, padding, kernel_width):
        # Pad/reshape before Sign, leaving an exact Sign -> Conv edge in ONNX.
        if padding:
            constants = torch.ones(
                x.shape[0], padding, dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, constants), dim=1)
        x = x.reshape(x.shape[0], 32, 1, kernel_width)
        y = F.conv2d(torch.sign(x), weight, None, stride=1, padding=0)
        return y[:, :, 0, 0] * scale.view(1, -1)

    def forward(self, raw_point_feature, previous_pose, previous_shape, previous_camera):
        aligned = self.maf_extractor.reduce_dim(raw_point_feature)
        combined = torch.cat(
            [aligned, previous_pose, previous_shape, previous_camera], dim=1
        )
        mean = combined.mean(dim=-1, keepdim=True)
        centered = combined - mean
        # torch.std defaults to the unbiased estimate. Write it explicitly:
        # the legacy ONNX exporter mis-converts aten::std for this graph.
        variance = (centered * centered).sum(dim=-1, keepdim=True) / float(
            combined.shape[-1] - 1
        )
        normalized = centered / (torch.sqrt(variance) + 1e-6)

        if self.bolt_bnn_linear_as_conv:
            first = self._binary_linear_as_conv(
                normalized, self.fc1_conv_weight, self.fc1_weight_scale,
                self.fc1_padding_value, self.fc1_kernel_width,
            )
            first = first + self.fc1_weight_bias.view(1, -1)
            first = self.regressor.drop1(first)
        elif self.stable_binary:
            first = F.linear(torch.sign(normalized), self.fc1_weight_bin, None)
            first = self.regressor.drop1(
                first * self.fc1_weight_scale.view(1, -1)
                + self.fc1_weight_bias.view(1, -1)
            )
        else:
            # Preserve the legacy accumulation order for exact PyTorch
            # partition validation. Deployment uses the static branch above.
            first = self.regressor.drop1(self.regressor.fc1(normalized))
        if self.bolt_bnn_linear_as_conv:
            second = self._binary_linear_as_conv(
                first, self.fc2_conv_weight, self.fc2_weight_scale,
                self.fc2_padding_value, self.fc2_kernel_width,
            )
            second = second + self.fc2_weight_bias.view(1, -1)
            second = self.regressor.drop2(second)
        elif self.stable_binary:
            # InferenceBiLinear equivalent with deterministic behavior around
            # zero. Tiny backend accumulation differences must not flip bits.
            if self.sign_epsilon > 0:
                binary_input = torch.where(
                    first > self.sign_epsilon,
                    torch.ones_like(first),
                    torch.where(first < -self.sign_epsilon, -torch.ones_like(first), torch.ones_like(first)),
                )
            else:
                # Match model_toB.ExportBiLinear exactly, including sign(0)=0.
                binary_input = torch.sign(first)
            second = F.linear(binary_input, self.fc2_weight_bin, None)
            second = self.regressor.drop2(
                second * self.fc2_weight_scale.view(1, -1)
                + self.fc2_weight_bias.view(1, -1)
            )
        else:
            second = self.regressor.drop2(self.regressor.fc2(first))
        mixed = (
            self.regressor.lamda1 * torch.matmul(first, self.regressor.hadamard)
            + self.regressor.lamda2 * second
        )
        pose = self.regressor.decpose(mixed) + previous_pose
        shape = self.regressor.decshape(mixed) + previous_shape
        camera = self.regressor.deccam(mixed) + previous_camera
        return pose, shape, camera


class FullPrecisionParameterStageWrapper(nn.Module):
    """Full-precision regression stage with the same deployment interface."""

    def __init__(self, maf_extractor, regressor):
        super().__init__()
        self.maf_extractor = maf_extractor
        self.regressor = regressor

    def forward(self, raw_point_feature, previous_pose, previous_shape, previous_camera):
        aligned = self.maf_extractor.reduce_dim(raw_point_feature)
        combined = torch.cat(
            [aligned, previous_pose, previous_shape, previous_camera], dim=1
        )
        hidden = self.regressor.drop1(self.regressor.fc1(combined))
        hidden = self.regressor.drop2(self.regressor.fc2(hidden))
        pose = self.regressor.decpose(hidden) + previous_pose
        shape = self.regressor.decshape(hidden) + previous_shape
        camera = self.regressor.deccam(hidden) + previous_camera
        return pose, shape, camera


def make_partitions(
    pymaf, stable_binary=False, sign_epsilon=1e-4, full_precision=False,
    bolt_bnn_linear_as_conv=False,
):
    stage_type = FullPrecisionParameterStageWrapper if full_precision else ParameterStageWrapper
    return (
        FeaturePyramidWrapper(pymaf).eval(),
        [
            (
                stage_type(pymaf.maf_extractor[index], pymaf.regressor[index])
                if full_precision
                else stage_type(
                    pymaf.maf_extractor[index], pymaf.regressor[index],
                    stable_binary=stable_binary, sign_epsilon=sign_epsilon,
                    bolt_bnn_linear_as_conv=bolt_bnn_linear_as_conv,
                )
            ).eval()
            for index in range(3)
        ],
    )
