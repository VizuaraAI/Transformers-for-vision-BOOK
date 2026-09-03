import pytest
import torch

from vision_bench.models import MODEL_SPECS, create_model, parameter_count


@pytest.mark.integration
@pytest.mark.parametrize("alias", MODEL_SPECS)
def test_model_instantiates_with_expected_parameter_scale(alias: str) -> None:
    spec = MODEL_SPECS[alias]
    model = create_model(alias, num_classes=100, pretrained=False)
    actual_millions = parameter_count(model) / 1e6
    assert actual_millions == pytest.approx(spec.expected_params_m, rel=0.08)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
