"""Time API model requests without changing settings, history or tool calls."""

from contextlib import asynccontextmanager

from pydantic_ai.models.wrapper import WrapperModel

from rpent.recording.narrated.instrumentation import phase_span


class TimedModel(WrapperModel):
    def __init__(self, wrapped, recorder, *, image_archive=None, episode_budget=None):
        super().__init__(wrapped)
        self.recorder = recorder
        self.image_archive = image_archive
        self.episode_budget = episode_budget

    def _archive_images(self, messages):
        if self.image_archive is not None and not getattr(self.wrapped, "captures_wire_images", False):
            self.image_archive.capture_model_request(messages, model=self.model_name)

    async def request(self, messages, model_settings, model_request_parameters):
        if self.episode_budget is not None and not getattr(self.wrapped, "handles_request_budget", False):
            self.episode_budget.reserve_request()
        self._archive_images(messages)
        with phase_span(self.recorder, "llm", model=self.model_name):
            return await self.wrapped.request(
                messages, model_settings, model_request_parameters
            )

    @asynccontextmanager
    async def request_stream(
        self, messages, model_settings, model_request_parameters, run_context=None
    ):
        if self.episode_budget is not None and not getattr(self.wrapped, "handles_request_budget", False):
            self.episode_budget.reserve_request()
        self._archive_images(messages)
        with phase_span(self.recorder, "llm", model=self.model_name):
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                yield stream
