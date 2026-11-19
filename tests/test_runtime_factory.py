

from rpent.runtime.factory import (
    authoritative_runtime_options,
)


def test_api_authoritative_runtime_needs_no_native_policy():
    assert authoritative_runtime_options("api") == {}
