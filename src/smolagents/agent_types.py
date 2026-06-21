# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent output types for smolagents.

This module defines the typed wrappers that agents return when producing outputs.
Each type dual-inherits from both :class:`AgentType` and a standard Python type so
that downstream code can treat the value as its native counterpart (``str``,
``PIL.Image.Image``, etc.) without any explicit conversion.

Available types:
    - :class:`AgentText`  — wraps a plain string
    - :class:`AgentImage` — wraps a ``PIL.Image.Image`` (also accepts file paths,
      raw ``bytes``, and ``torch.Tensor`` / ``numpy.ndarray``)
    - :class:`AgentAudio` — wraps a ``torch.Tensor`` audio waveform (also accepts
      file paths and ``(samplerate, array)`` tuples)

Helper functions:
    - :func:`handle_agent_input_types` — unwraps :class:`AgentType` wrappers in
      ``*args`` / ``**kwargs`` before they are passed to a tool's ``forward()``.
    - :func:`handle_agent_output_types` — wraps a raw tool return value in the
      appropriate :class:`AgentType` subclass.
"""

import logging
import os
import pathlib
import tempfile
import uuid
from io import BytesIO
from typing import Any

import PIL.Image
import requests

from .utils import _is_package_available


logger = logging.getLogger(__name__)


class AgentType:
    """
    Abstract base class for types that can be returned by agents.

    These objects serve three purposes:

    - They behave as the type they're meant to represent, e.g., a string for
      text, a ``PIL.Image.Image`` for images.
    - They can be stringified via ``str(obj)`` to return a string representation
      (often a filesystem path for binary types).
    - They display correctly in IPython notebooks / Colab / Jupyter via
      ``_ipython_display_``.

    Concrete subclasses must override :meth:`to_raw` and :meth:`to_string`.
    """

    def __init__(self, value: Any) -> None:
        self._value = value

    def __str__(self) -> str:
        return self.to_string()

    def to_raw(self) -> Any:
        """Return the underlying raw Python object (e.g. ``PIL.Image.Image``).

        The base implementation logs a warning and returns the stored value
        as-is; subclasses should override this to return the appropriate type.
        """
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return self._value

    def to_string(self) -> str:
        """Return a string representation of this value.

        For binary types such as :class:`AgentImage` and :class:`AgentAudio`
        this is a filesystem path to a serialized file.  For text types it is
        the text itself.  The base implementation logs a warning and falls back
        to ``str(self._value)``.
        """
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return str(self._value)


class AgentText(AgentType, str):
    """
    Text type returned by the agent.  Behaves as a plain Python ``str``.

    Example::

        result = AgentText("Hello, world!")
        assert isinstance(result, str)  # True
        print(result)                   # Hello, world!
    """

    def to_raw(self) -> str:
        """Return the underlying string value."""
        return self._value

    def to_string(self) -> str:
        """Return the string representation (the value itself)."""
        return str(self._value)


class AgentImage(AgentType, PIL.Image.Image):
    """
    Image type returned by the agent.  Behaves as a ``PIL.Image.Image``.

    The constructor accepts multiple source formats and normalises them
    internally:

    - ``PIL.Image.Image`` — used directly as the raw image.
    - ``bytes`` — decoded via ``PIL.Image.open``.
    - ``str`` / ``pathlib.Path`` — treated as a filesystem path; the image is
      loaded lazily on first access.
    - ``torch.Tensor`` / ``numpy.ndarray`` — stored as a tensor; converted to
      a PIL image on first access.
    - Another :class:`AgentImage` — copies the internal state.

    Args:
        value: Source image in any of the supported formats listed above.

    Raises:
        TypeError: If ``value`` is not one of the supported types.
    """

    def __init__(self, value: Any) -> None:
        AgentType.__init__(self, value)
        PIL.Image.Image.__init__(self)

        self._path = None
        self._raw = None
        self._tensor = None

        if isinstance(value, AgentImage):
            self._raw, self._path, self._tensor = value._raw, value._path, value._tensor
        elif isinstance(value, PIL.Image.Image):
            self._raw = value
        elif isinstance(value, bytes):
            self._raw = PIL.Image.open(BytesIO(value))
        elif isinstance(value, (str, pathlib.Path)):
            self._path = value
        else:
            try:
                import torch

                if isinstance(value, torch.Tensor):
                    self._tensor = value
                import numpy as np

                if isinstance(value, np.ndarray):
                    self._tensor = torch.from_numpy(value)
            except ModuleNotFoundError:
                pass

        if self._path is None and self._raw is None and self._tensor is None:
            raise TypeError(f"Unsupported type for {self.__class__.__name__}: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None) -> None:
        """
        Display the image correctly in an IPython notebook (Jupyter, Colab, etc.).
        """
        from IPython.display import Image, display

        display(Image(self.to_string()))

    def to_raw(self) -> PIL.Image.Image:
        """
        Return the raw ``PIL.Image.Image`` representation.

        The image is decoded or converted from the internal storage format
        (path, bytes, or tensor) on first call and then cached.
        """
        if self._raw is not None:
            return self._raw

        if self._path is not None:
            self._raw = PIL.Image.open(self._path)
            return self._raw

        if self._tensor is not None:
            import numpy as np

            array = self._tensor.cpu().detach().numpy()
            return PIL.Image.fromarray((255 - array * 255).astype(np.uint8))

    def to_string(self) -> str:
        """
        Return a filesystem path to a PNG serialisation of this image.

        If the image was originally loaded from a path, that path is returned
        unchanged.  Otherwise the image is written to a temporary file and its
        path is returned (and cached for subsequent calls).
        """
        if self._path is not None:
            return self._path

        if self._raw is not None:
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")
            self._raw.save(self._path, format="png")
            return self._path

        if self._tensor is not None:
            import numpy as np

            array = self._tensor.cpu().detach().numpy()

            # There is likely simpler than load into image into save
            img = PIL.Image.fromarray((255 - array * 255).astype(np.uint8))

            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")
            img.save(self._path, format="png")

            return self._path

    def save(self, output_bytes: Any, format: str | None = None, **params: Any) -> None:
        """Save the image to a file-like object or path.

        This is a thin wrapper around ``PIL.Image.Image.save`` that ensures the
        raw PIL image is obtained first via :meth:`to_raw`.

        Args:
            output_bytes: Destination — a writable file-like object or a
                filesystem path accepted by ``PIL.Image.save``.
            format: Image format string (e.g. ``"PNG"``, ``"JPEG"``) as
                accepted by ``PIL.Image.save``.  If ``None``, PIL infers the
                format from the file extension.
            **params: Additional keyword arguments forwarded to
                ``PIL.Image.save``.
        """
        img = self.to_raw()
        img.save(output_bytes, format=format, **params)


class AgentAudio(AgentType, str):
    """
    Audio type returned by the agent.

    Wraps an audio waveform and exposes it as a ``torch.Tensor``.  The
    constructor accepts several input formats:

    - ``str`` / ``pathlib.Path`` — filesystem path or URL to an audio file
      readable by ``soundfile``.
    - ``torch.Tensor`` — raw waveform tensor.
    - ``tuple`` — a ``(samplerate, array)`` pair where ``array`` is a
      ``numpy.ndarray`` or any sequence accepted by ``torch.tensor``.

    Args:
        value: Audio source in any of the supported formats.
        samplerate: Sample rate in Hz.  Ignored when ``value`` is a tuple
            (the tuple's first element is used instead).  Defaults to
            ``16_000``.

    Raises:
        ModuleNotFoundError: If ``soundfile`` or ``torch`` are not installed.
        ValueError: If ``value`` is not a supported type.
    """

    def __init__(self, value: Any, samplerate: int = 16_000) -> None:
        if not _is_package_available("soundfile") or not _is_package_available("torch"):
            raise ModuleNotFoundError(
                "Please install 'audio' extra to use AgentAudio: `pip install 'smolagents[audio]'`"
            )
        import numpy as np
        import torch

        super().__init__(value)

        self._path = None
        self._tensor = None

        self.samplerate = samplerate
        if isinstance(value, (str, pathlib.Path)):
            self._path = value
        elif isinstance(value, torch.Tensor):
            self._tensor = value
        elif isinstance(value, tuple):
            self.samplerate = value[0]
            if isinstance(value[1], np.ndarray):
                self._tensor = torch.from_numpy(value[1])
            else:
                self._tensor = torch.tensor(value[1])
        else:
            raise ValueError(f"Unsupported audio type: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None) -> None:
        """
        Display the audio correctly in an IPython notebook (Jupyter, Colab, etc.).
        """
        from IPython.display import Audio, display

        display(Audio(self.to_string(), rate=self.samplerate))

    def to_raw(self) -> Any:
        """
        Return the raw ``torch.Tensor`` waveform.

        If the audio was loaded from a path or URL, the file is read via
        ``soundfile`` on first call and the result is cached.
        """
        import soundfile as sf

        if self._tensor is not None:
            return self._tensor

        import torch

        if self._path is not None:
            if "://" in str(self._path):
                response = requests.get(self._path)
                response.raise_for_status()
                tensor, self.samplerate = sf.read(BytesIO(response.content))
            else:
                tensor, self.samplerate = sf.read(self._path)
            self._tensor = torch.tensor(tensor)
            return self._tensor

    def to_string(self) -> str:
        """
        Return a filesystem path to a WAV serialisation of this audio.

        If the audio was originally loaded from a path, that path is returned
        unchanged.  Otherwise the waveform is written to a temporary ``.wav``
        file and its path is returned (and cached for subsequent calls).
        """
        import soundfile as sf

        if self._path is not None:
            return self._path

        if self._tensor is not None:
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".wav")
            sf.write(self._path, self._tensor, samplerate=self.samplerate)
            return self._path


_AGENT_TYPE_MAPPING = {"string": AgentText, "image": AgentImage, "audio": AgentAudio}


def handle_agent_input_types(*args: Any, **kwargs: Any) -> tuple[tuple, dict]:
    """Unwrap :class:`AgentType` wrappers from tool call arguments.

    Before a tool's ``forward()`` method is called, any :class:`AgentType`
    instances in ``args`` or ``kwargs`` are replaced by their raw underlying
    values (via :meth:`AgentType.to_raw`).  Non-``AgentType`` values are
    passed through unchanged.

    Args:
        *args: Positional arguments to the tool.
        **kwargs: Keyword arguments to the tool.

    Returns:
        A ``(args, kwargs)`` tuple with all :class:`AgentType` values
        replaced by their raw counterparts.
    """
    args = tuple((arg.to_raw() if isinstance(arg, AgentType) else arg) for arg in args)
    kwargs = {k: (v.to_raw() if isinstance(v, AgentType) else v) for k, v in kwargs.items()}
    return args, kwargs


def handle_agent_output_types(output: Any, output_type: str | None = None) -> Any:
    """Wrap a raw tool return value in the appropriate :class:`AgentType`.

    When ``output_type`` is provided and matches a known type name
    (``"string"``, ``"image"``, or ``"audio"``), the output is wrapped in the
    corresponding :class:`AgentType` subclass.  Otherwise the type is inferred
    from the runtime type of ``output``:

    - ``str``              → :class:`AgentText`
    - ``PIL.Image.Image``  → :class:`AgentImage`
    - ``torch.Tensor``     → :class:`AgentAudio`
    - anything else        → returned as-is

    Args:
        output: The raw value returned by the tool's ``forward()`` method.
        output_type: Optional declared output type name from the tool's
            ``output_type`` class attribute.  One of ``"string"``,
            ``"image"``, or ``"audio"``.

    Returns:
        The output wrapped in the appropriate :class:`AgentType`, or the
        original value if no wrapping is applicable.
    """
    if output_type in _AGENT_TYPE_MAPPING:
        # If the class has defined outputs, we can map directly according to the class definition
        decoded_outputs = _AGENT_TYPE_MAPPING[output_type](output)
        return decoded_outputs

    # If the class does not have defined output, then we map according to the type
    if isinstance(output, str):
        return AgentText(output)
    if isinstance(output, PIL.Image.Image):
        return AgentImage(output)
    try:
        import torch

        if isinstance(output, torch.Tensor):
            return AgentAudio(output)
    except ModuleNotFoundError:
        pass
    return output


__all__ = ["AgentType", "AgentImage", "AgentText", "AgentAudio"]
