from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RegisterRequest(_message.Message):
    __slots__ = ["user_id", "video_path"]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    VIDEO_PATH_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    video_path: str
    def __init__(self, user_id: _Optional[str] = ..., video_path: _Optional[str] = ...) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = ["success"]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...
