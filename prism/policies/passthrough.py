"""The `none` policy: the model sees the raw output, unchanged. Used by replay and tests."""

from prism.policy import register


@register("none")
class PassthroughPolicy:
    requires: set[str] = set()

    def apply(self, ev, store) -> str:
        return ev.output
