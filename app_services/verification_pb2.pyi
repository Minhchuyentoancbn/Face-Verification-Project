from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class VerificationRequest(_message.Message):
    __slots__ = ["user_id", "video_path"]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    VIDEO_PATH_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    video_path: str
    def __init__(self, user_id: _Optional[str] = ..., video_path: _Optional[str] = ...) -> None: ...

class VerificationResponse(_message.Message):
    __slots__ = ["similarity", "verified"]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    VERIFIED_FIELD_NUMBER: _ClassVar[int]
    similarity: float
    verified: bool
    def __init__(self, verified: bool = ..., similarity: _Optional[float] = ...) -> None: ...
