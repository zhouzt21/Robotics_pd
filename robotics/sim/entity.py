import sapien
import numpy as np
from sapien import Scene, physx
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, TYPE_CHECKING, Optional, cast, Type, TypeVar
if TYPE_CHECKING:
    from .simulator import Simulator

T = TypeVar('T')

class Entity(ABC):
    __children__: Dict[str, "Entity"]

    @abstractmethod
    def _load(self, world: "Simulator"):
        ...

    def _get_sapien_entity(self) -> List[sapien.Entity | physx.PhysxArticulation |Any]:
        return []

    def _find(self, name: str): 
        raise ValueError(f"Cannot find {name} in the entity {self}")

    def _clear(self):
        # additional for clearing the element
        # note that the sapien entity is cleared by the simulator
        ...

    def _before_simulation_step(self, *args, **kwargs):
        pass

    def add_subentity(self, name: str, child: "Entity"):
        if not hasattr(self, '__children__'):
            object.__setattr__(self, '__children__', {})
        self.__children__[name] = child

    def children(self):
        if not hasattr(self, '__children__'):
            return {}
        return self.__children__

    def find(self, uid: str):
        if '/' in uid:
            k, uid = uid.split('/', 1)
            return self.children()[k].find(uid)
        else:
            if hasattr(self, '__children__') and uid in self.__children__:
                return self.__children__[uid]
            return self._find(uid)


    def load(self, world: "Simulator"):
        world.load_entity(self)
        for v in self.children().values():
            v.load(world)

    def before_simulation_step(self, *args, **kwargs):
        self._before_simulation_step(*args, **kwargs)
        for v in self.children().values():
            v.before_simulation_step(*args, **kwargs)

    def clear(self):
        self._clear()
        for v in self.children().values():
            v.clear()

    @property
    def observation_space(self) -> Any:
        return None

    def get_sapien_obj_type(self, cls: Type[T]) -> list[T]:
        output = [i for i in self._get_sapien_entity() if isinstance(i, cls)]
        for v in self.children().values():
            output.extend(v.get_sapien_obj_type(cls))
        return output

    def get_articulations(self) -> List[physx.PhysxArticulation]:
        output = [i for i in self._get_sapien_entity() if isinstance(i, physx.PhysxArticulation)]
        for v in self.children().values():
            output.extend(v.get_articulations())
        return output

    def get_actors(self) -> List[sapien.Entity]:
        output = [i for i in self._get_sapien_entity() if isinstance(i, sapien.Entity)]
        for v in self.children().values():
            output.extend(v.get_actors())
        return output

    def get_meshes(self, exclude=()):
        from robotics.utils.sapien_utils import get_actor_mesh, get_articulation_meshes, merge_meshes
        v = self
        meshes = []
        for articulation in v.get_articulations():
            articulation_mesh = merge_meshes(get_articulation_meshes(articulation))
            if articulation_mesh:
                meshes.append(articulation_mesh)

        # print("exclude", exclude)
        for actor in v.get_actors():
            if actor.get_name() in exclude:
                # print('Excluding', exclude, actor.get_name())
                continue

            if actor.get_pose().p[2] > 100:
                # Ignore the actor that is too high
                continue
            actor_mesh = get_actor_mesh(actor)

            if actor_mesh:
                # print('not excluded', actor.get_name())
                meshes.append(actor_mesh)
        return merge_meshes(meshes)

    def get_pcds(self, num_points: int = int(1e5), exclude=()) -> Optional[np.ndarray]:
        scene_mesh = self.get_meshes(exclude)
        if scene_mesh is None:
            return None
        return cast(np.ndarray, scene_mesh.sample(num_points))



class Composite(Entity):
    def __init__(self, uid: str='', **components: Union[Entity,Dict]) -> None:
        outputs = {}
        for k, v in components.items():
            child_uid = uid + '/' + k
            if isinstance(v, dict):
                v = Composite(child_uid, **v)
            outputs[k] = v
        self.__children__ = outputs

    def _load(self, world: "Simulator"):
        pass

    def __iter__(self):
        return iter(self.__children__.values())

    def items(self):
        return self.__children__.items()

    def __getitem__(self, key):
        return self.__children__[key]
