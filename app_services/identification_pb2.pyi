from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class IdentificationRequest(_message.Message):
    __slots__ = ["video_path"]
    VIDEO_PATH_FIELD_NUMBER: _ClassVar[int]
    video_path: str
    def __init__(self, video_path: _Optional[str] = ...) -> None: ...

class IdentificationResponse(_message.Message):
    __slots__ = ["user_id"]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    def __init__(self, user_id: _Optional[str] = ...) -> None: ...
