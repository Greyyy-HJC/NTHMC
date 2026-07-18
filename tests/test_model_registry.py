import pytest

from nthmc.u1.models import LocalNet as JaxU1Net
from nthmc.u1.models import choose_model as choose_jax_u1_model
from nthmc.u1.training_models import LocalNet as TorchU1Net
from nthmc.u1.training_models import choose_model as choose_torch_u1_model
from nthmc.u2.models import LocalNet as JaxU2Net
from nthmc.u2.models import choose_model as choose_jax_u2_model
from nthmc.u2.training_models import LocalNet as TorchU2Net
from nthmc.u2.training_models import choose_model as choose_torch_u2_model


@pytest.mark.parametrize(
    ("choose_model", "expected"),
    [
        (choose_jax_u1_model, JaxU1Net),
        (choose_torch_u1_model, TorchU1Net),
        (choose_jax_u2_model, JaxU2Net),
        (choose_torch_u2_model, TorchU2Net),
    ],
)
def test_base_is_the_only_registered_model(choose_model, expected) -> None:
    assert choose_model("base") is expected
    with pytest.raises(ValueError, match="Invalid .* model tag"):
        choose_model("unregistered")
