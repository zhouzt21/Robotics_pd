import os
import transforms3d
from typing import Tuple, Optional, Dict, cast, List
from pathlib import Path
import sapien
import numpy as np
from robotics.sim import Simulator
from ..environ import EnvironBase, EnvironConfig
from sapien import physx

from robotics.utils.path import PACKAGE_ASSET_DIR
import json
from robotics import Pose


class YCBConfig(EnvironConfig):
    N: int = 5
    scene_id: int = 0 # control the random seed
    bbox: Tuple[float, float, float, float] = (-1., -1., 1., 1.)


YCB_PATH = Path(PACKAGE_ASSET_DIR) / 'data' / 'mani_skill2_ycb'



from robotics.utils.sapien_utils import get_actor_mesh


def build_actor_ycb(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material = None, # TODO: type
    density=1000,
    #root_dir=ASSET_DIR / "mani_skill2_ycb",
):
    builder = scene.create_actor_builder()
    model_dir = Path(PACKAGE_ASSET_DIR) / 'data' / 'mani_skill2_ycb' / "models" / model_id

    collision_file = str(model_dir / "collision.obj")
    builder.add_multiple_convex_collisions_from_file(
        filename=collision_file,
        scale=(scale,) * 3,
        material=physical_material, # type: ignore
        density=density,
        decomposition='coacd'
    )

    visual_file = str(model_dir / "textured.obj")
    builder.add_visual_from_file(filename=visual_file, scale=(scale,) * 3)

    actor = builder.build()
    actor.name = model_id

    # actor.set_damping(0.1, 0.1)
    return actor


class YCBClutter(EnvironBase):
    _actors: List[sapien.Entity]
    def __init__(self, config: Optional[YCBConfig]=None, keyword: Optional[str]=None) -> None:
        config = config or YCBConfig()
        self.config = config
        if not os.path.exists(YCB_PATH):
            raise RuntimeError(
                f"Please download the YCB dataset by running "
                f"`python -m mani_skill2.utils.download_asset PickClutterYCB-v0 -o {PACKAGE_ASSET_DIR}/data`."
            )
        with open(Path(PACKAGE_ASSET_DIR) / 'data' / 'mani_skill2_ycb' / "info_pick_v0.json") as f:
            self.model_db: Dict[str, Dict] = json.loads(f.read())
        if keyword is not None:
            self.model_db = {k: v for k, v in self.model_db.items() if keyword in k}
        self._actors = []

    def _load(self, world: Simulator, scene_id: Optional[int]=None):
        # return super().load(world)
        if len(self._actors) > 0:
            for k in self._actors:
                # TODO: remove actor
                #world._scene.remove_actor(k)
                pass
            self._actors = []
        print('load ycb clutter env')

        self._model_id = []
        state = None
        scene_id = scene_id or self.config.scene_id
        if scene_id is not None:
            state = np.random.get_state()
            np.random.seed(scene_id)

        #TODO: use sim_utils.load_actors_without_collision

        bbox = np.array(self.config.bbox)
        for i in range(self.config.N):

            model_id = np.random.choice(list(self.model_db.keys()))
            actor, obj_comp = self._load_model(world._scene, model_id, model_scale=1.) # np.random.random() * 1 + 1
            self._actors.append(actor)
            self._model_id.append([list(self.model_db.keys()).index(model_id), 1.]) # model scale

            mesh = get_actor_mesh(actor)
            assert mesh is not None
            pcd_obj = cast(np.ndarray, mesh.sample(1000))
            pcd = world.gen_scene_pcd(200000)
            from scipy.spatial import cKDTree
            tree = cKDTree(pcd)


            while True:
                x, y = np.random.uniform(bbox[:2], bbox[2:])
                orientation = np.random.uniform(-np.pi, np.pi)
                pose = Pose(np.array([x, y, 0.2]), transforms3d.euler.euler2quat(0, 0, orientation))

                pcd_obj2 = np.concatenate([pcd_obj, np.ones((pcd_obj.shape[0], 1))], axis=1)
                pcd2 = pcd_obj2 @ (cast(np.ndarray, pose.to_transformation_matrix().T))

                distances, _ = tree.query(pcd2[..., :3], k=1)
                if distances.min() > .05:
                    break

            actor.set_pose(pose)
        np.random.set_state(state)

    def _load_model(self, scene: sapien.Scene, model_id, model_scale=1.0):
        density = self.model_db[model_id].get("density", 1000)
        obj = build_actor_ycb(
            model_id,
            scene,
            scale=model_scale,
            density=density,
        )
        obj.name = model_id
        from robotics.utils.sapien_utils import get_rigid_dynamic_component
        obj_comp = get_rigid_dynamic_component(obj)
        assert obj_comp is not None

        obj_comp.set_linear_damping(0.1)
        obj_comp.set_angular_damping(0.1)
        return obj, obj_comp

    def _get_sapien_entity(self):
        return self._actors