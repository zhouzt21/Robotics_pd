from dataclasses import dataclass
from typing_extensions import dataclass_transform

@dataclass_transform()
@dataclass
class Config:
    def __init_subclass__(cls, *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)
        if '__dataclass_fields__' not in cls.__dict__:
            cls = dataclass(cls)