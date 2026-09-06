"""Formula, shape, gradient, serialization, and optional CUDA checks."""

import io

import pytest
import torch

from modules import AIC, GCE, MSA


@pytest.fixture(autouse=True)
def deterministic_inputs():
    torch.manual_seed(7)
    torch.set_num_threads(2)


@pytest.mark.parametrize('channels', [1, 7, 64])
def test_aic_bounded_channel_gain(channels):
    module = AIC(channels)
    x = torch.randn(2, channels, 9, 13)
    gain = module.calibration(x)
    assert gain.shape == (2, channels, 1, 1)
    assert torch.all((gain >= 1) & (gain <= 2))
    torch.testing.assert_close(module(x), x * gain)


def test_aic_zero_logits_gives_residual_gain():
    module = AIC(8)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    x = torch.randn(2, 8, 5, 11)
    torch.testing.assert_close(module(x), 1.5 * x)


def test_gce_axes_and_multiplicative_fusion():
    module = GCE(8)
    x = torch.randn(2, 8, 5, 11)
    mc, mw, mh = module.attention(x)
    assert mc.shape == (2, 8, 1, 1)
    assert mw.shape == (2, 8, 1, 11)
    assert mh.shape == (2, 8, 5, 1)
    torch.testing.assert_close(module(x), x * mc * mw * mh)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    torch.testing.assert_close(module(x), x / 8)


@pytest.mark.parametrize('module_type', [AIC, GCE])
def test_single_scale_gradients_and_state_roundtrip(module_type):
    module = module_type(8)
    x = torch.randn(2, 8, 7, 11, requires_grad=True)
    output = module(x)
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    state = io.BytesIO()
    torch.save(module.state_dict(), state)
    state.seek(0)
    restored = module_type(8)
    restored.load_state_dict(torch.load(state, weights_only=True))
    torch.testing.assert_close(module(x.detach()), restored(x.detach()))


@pytest.mark.parametrize('target', [0, 1, 2])
def test_msa_odd_rectangular_scales_and_all_input_gradients(target):
    # Deliberately non-integer resolution ratios catch hard-coded scale factors.
    inputs = [torch.randn(2, c, h, w, requires_grad=True)
              for c, h, w in [(8, 17, 25), (16, 9, 13), (32, 5, 7)]]
    order = [target] + [i for i in range(3) if abs(i - target) == 1]
    features = [inputs[i] for i in order]
    module = MSA([f.shape[1] for f in features])
    copies = [f.detach().clone() for f in features]
    output = module(features)
    assert output.shape == features[0].shape
    output.square().mean().backward()
    for feature, before in zip(features, copies):
        torch.testing.assert_close(feature, before)
        assert feature.grad is not None and torch.isfinite(feature.grad).all()
        assert feature.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_msa_independent_sigmoid_weights_and_residual():
    module = MSA([1, 1, 1])
    with torch.no_grad():
        for projection in module.projections:
            projection.weight.fill_(1)
        for gate in module.gates:
            gate[0].weight.zero_()
            gate[0].bias.zero_()
    current = torch.ones(1, 1, 7, 11)
    neighbors = [torch.full((1, 1, 3, 5), 2.), torch.full((1, 1, 13, 21), 4.)]
    # sigmoid(0)=0.5 for EACH neighbor: 1 + 0.5*2 + 0.5*4 = 4.
    torch.testing.assert_close(module([current, *neighbors]), torch.full_like(current, 4.))


def test_msa_state_roundtrip():
    module, restored = MSA([8, 16]), MSA([8, 16])
    restored.load_state_dict(module.state_dict())
    inputs = [torch.randn(1, 8, 9, 13), torch.randn(1, 16, 5, 7)]
    torch.testing.assert_close(module(inputs), restored(inputs))


@pytest.mark.parametrize('module_type', [AIC, GCE])
def test_invalid_channels(module_type):
    with pytest.raises(ValueError):
        module_type(0)
    with pytest.raises(ValueError):
        module_type(8, reduction=0)


def test_msa_invalid_inputs():
    with pytest.raises(ValueError):
        MSA([8])
    with pytest.raises(ValueError):
        MSA([8, 0])
    with pytest.raises(ValueError):
        MSA([8, 16])([torch.zeros(1, 8, 4, 4)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_mixed_precision_all_modules():
    x = torch.randn(2, 8, 16, 24, device='cuda', requires_grad=True)
    neighbor = torch.randn(2, 16, 8, 12, device='cuda', requires_grad=True)
    aic, gce, msa = AIC(8).cuda(), GCE(8).cuda(), MSA([8, 16]).cuda()
    with torch.autocast('cuda', dtype=torch.float16):
        output = msa([gce(aic(x)), neighbor])
        loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for module in (aic, gce, msa) for p in module.parameters())
