from __future__ import annotations

import aassr_v2
import pytest
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.integrated_agent import (
    build_full_aassr_core as legacy_v040_full_builder,
    build_pentest_aassr_core as legacy_v040_builder,
)


def test_package_default_pentest_builder_is_current_generation() -> None:
    assert aassr_v2.build_pentest_aassr_core is build_current_pentest_aassr_core
    assert aassr_v2.build_legacy_v040_pentest_aassr_core is legacy_v040_builder
    assert aassr_v2.build_pentest_aassr_core is not legacy_v040_builder


def test_package_public_api_keeps_current_and_legacy_builders_unambiguous() -> None:
    assert "build_pentest_aassr_core" in aassr_v2.__all__
    assert "build_legacy_v040_full_aassr_core" in aassr_v2.__all__
    assert "build_legacy_v040_pentest_aassr_core" in aassr_v2.__all__
    assert "build_full_aassr_core" not in aassr_v2.__all__
    assert "build_full_aassr_core" not in dir(aassr_v2)

    with pytest.warns(DeprecationWarning, match="legacy v0.4 compatibility alias"):
        compatibility_builder = aassr_v2.build_full_aassr_core
    assert compatibility_builder is legacy_v040_full_builder
    assert aassr_v2.build_pentest_aassr_core is build_current_pentest_aassr_core
    assert aassr_v2.build_pentest_aassr_core is not legacy_v040_builder
