from __future__ import annotations

import aassr_v2
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.integrated_agent import build_pentest_aassr_core as legacy_v040_builder


def test_package_default_pentest_builder_is_current_generation() -> None:
    assert aassr_v2.build_pentest_aassr_core is build_current_pentest_aassr_core
    assert aassr_v2.build_legacy_v040_pentest_aassr_core is legacy_v040_builder
    assert aassr_v2.build_pentest_aassr_core is not legacy_v040_builder
