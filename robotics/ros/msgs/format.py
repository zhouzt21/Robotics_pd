from typing import Type, Any, TypeVar, Tuple, Iterator


class Format:
    dtype: Any

    def to_msg(self):
        raise NotImplementedError

    @classmethod
    def from_msg(cls, msg: Any):
        raise NotImplementedError

    @classmethod
    def add_stamp(cls, msg: Any, stamp: Any):
        msg.header.stamp = stamp
        return msg

        
C = TypeVar("C", bound=Format)
FormatType = Type[C]


class ROSMsg(Format):
    @classmethod
    def to_msg(cls, msg):
        return msg

    @classmethod
    def from_msg(cls, msg: Any):
        return msg


class Struct(Format):
    dtype: Any = None

    @classmethod
    def __get_subfields__(cls) -> Iterator[Tuple[str, Any]]:
        for k, v in cls.__annotations__.items():
            if k!='dtype':
                yield k, v

    @classmethod
    def to_msg(cls, **kwargs):
        output = {}
        for k, v in cls.__get_subfields__():
            if k not in kwargs:
                raise ValueError(f"Missing field {k} in {cls}")
            ans = kwargs[k]
            if issubclass(v, Format):
                ans = v.to_msg(ans)
            output[k] = ans
        assert len(kwargs) == len(output), f"Extra fields in {cls}: {set(kwargs.keys()) - set(output.keys())} for {kwargs}"
        return output
            

    @classmethod
    def from_msg(cls, msg: Any):
        output = {}
        for k, v in cls.__get_subfields__():
            ans = getattr(msg, k)
            if issubclass(v, Format):
                ans = v.from_msg(ans)
            output[k] = ans
        return output


    @classmethod
    def add_stamp(cls, msg: Any, stamp: Any):
        if hasattr(msg, 'header'):
            msg.header.stamp = stamp
        else:
            for k, v in cls.__get_subfields__():
                if issubclass(v, Format):
                    setattr(msg, k, v.add_stamp(getattr(msg, k), stamp))
        return msg


# def from_msg(msg_type: FormatType, msg: Any):
#     if isinstance(msg_type, tuple):
#         return tuple(from_msg(t, m) for t, m in zip(msg_type, msg))
#     return msg_type.from_msg(msg)

    
# def to_msg(msg_type: FormatType, *args, **kwargs):
#     if isinstance(msg_type, tuple):
#         assert len(msg_type) == len(args)
#         assert len(kwargs) == 0
#         return tuple(to_msg(t, *a) for t, a in zip(msg_type, args))
#     return msg_type.to_msg(*args, **kwargs)

# def add_stamp(msg_type: FormatType, msg: Any, stamp: Any):
#     if isinstance(msg_type, tuple):
#         return tuple(add_stamp(t, m, stamp) for t, m in zip(msg_type, msg))
#     return msg_type.add_stamp(msg, stamp)

    
# def get_dtype(msg_type: FormatType) -> Any:
#     if isinstance(msg_type, tuple):
#         return tuple(get_dtype(t) for t in msg_type)
#     return msg_type.dtype