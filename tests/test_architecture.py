from __future__ import annotations

import worldlab.core as core
from worldlab.world_models import WorldModel


def test_world_model_belongs_to_models_not_core() -> None:
    assert not hasattr(core, "WorldModel")
    assert WorldModel.__module__ == "worldlab.world_models.base"
