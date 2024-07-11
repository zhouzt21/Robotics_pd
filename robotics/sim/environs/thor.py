from __future__ import annotations
try:
    import mani_skill
except ImportError as e:
    raise ImportError("Thor env requires ManiSkill3 to be installed, see https://maniskill.readthedocs.io/en/dev/user_guide/getting_started/installation.html")

import numpy as np
from tqdm import tqdm
import os.path as osp
import json
import sapien
from sapien import physx
import transforms3d
from pathlib import Path

from dataclasses import field
import sapien
from ..environ import EnvironBase, EnvironConfig
from robotics.utils.path import PACKAGE_ASSET_DIR
from robotics.utils.sftp import download_if_not_exists
import mani_skill.envs # NOTE: must import mani_skill.envs first to avoid circular import
from mani_skill.utils.scene_builder.ai2thor.scene_builder import load_ai2thor_metadata, SCENE_SOURCE_TO_DATASET, SceneConfig, ALL_SCENE_CONFIGS, DATASET_CONFIG_DIR, ASSET_DIR, WORKING_OBJS
from robotics.sim import Simulator


class ThorEnvConfig(EnvironConfig):
    convex_decomposition: str = "coacd"
    scene_dataset: str = "ArchitecTHOR"
    index: int = 0


class ThorEnv(EnvironBase):
    config: ThorEnvConfig
    has_ground = True
    def __init__(self, config: ThorEnvConfig | None = None):
        config = config or ThorEnvConfig()
        super().__init__(config)
        # print(os.path.abspath(os.path.join(PACKAGE_ASSET_DIR, '../../third_party/ManiSkill3/data/scene_datasets')))
        # download_if_not_exists(os.path.join(PACKAGE_ASSET_DIR, '../../third_party/ManiSkill3/data/scene_datasets'), 'scene_datasets')

        try:

            global OBJECT_SEMANTIC_ID_MAPPING, SEMANTIC_ID_OBJECT_MAPPING, MOVEABLE_OBJECT_IDS
            (
                OBJECT_SEMANTIC_ID_MAPPING,
                SEMANTIC_ID_OBJECT_MAPPING,
                MOVEABLE_OBJECT_IDS,
            ) = load_ai2thor_metadata()
        except FileNotFoundError as e:
            filename = e.filename
            raise FileNotFoundError("You can run python -m robotics.sim.environs.download <HF_TOKEN> to download the ai2thor-hab dataset first to {} (HF_TOKEN is your token in huggingface). But it is more recommended to use git lfs to download it (https://huggingface.co/docs/hub/en/datasets-downloading).".format(filename))
        self.scene_dataset = config.scene_dataset
        self._scene_configs: list[SceneConfig] = []
        if self.scene_dataset not in ALL_SCENE_CONFIGS:
            dataset_path = SCENE_SOURCE_TO_DATASET[self.scene_dataset].metadata_path
            with open(osp.join(DATASET_CONFIG_DIR, dataset_path)) as f:
                scene_jsons = json.load(f)["scenes"]
            self._scene_configs += [
                SceneConfig(config_file=scene_json, source=self.scene_dataset)
                for scene_json in scene_jsons
            ]
            ALL_SCENE_CONFIGS[self.scene_dataset] = self._scene_configs
        else:
            self._scene_configs = ALL_SCENE_CONFIGS[self.scene_dataset]

        self._scene_navigable_positions = [None] * len(self._scene_configs)
        self.actor_default_poses: list[tuple[sapien.Entity, sapien.Pose]] = []

        self.scene_idx = 0

    # def _load(self, world: Simulator):
    #     print('load thor env')
    #     scene_cfg: SceneConfig = self._scene_configs[self.config.index]

    #     dataset = SCENE_SOURCE_TO_DATASET[scene_cfg.source]
    #     scene_adapter = dataset.adapter(convex_decomposition=self.config.convex_decomposition) # get adapter to adapt scene metadata from this source
    #     with open(osp.join(dataset.dataset_path, scene_cfg.config_file), "rb") as f:
    #         config_file = json.load(f)
    #     scene_adapter.create_ms2_scene(config_file, world._scene)
    #     self.scene_objects = scene_adapter.get_scene_objects()

        
    def _load(
        self, world: Simulator,
    ):
        scene = world._scene
        self._scene_objects: dict[str, sapien.Entity] = dict()
        self._movable_objects: dict[str, sapien.Entity] = dict()

        scene_cfg = self._scene_configs[self.scene_idx]

        dataset = SCENE_SOURCE_TO_DATASET[scene_cfg.source]
        with open(osp.join(dataset.dataset_path, scene_cfg.config_file), "rb") as f:
            scene_json = json.load(f)

        bg_path = str(
            Path(ASSET_DIR)
            / "scene_datasets/ai2thor/ai2thor-hab/assets"
            / f"{scene_json['stage_instance']['template_name']}.glb"
        )
        builder = scene.create_actor_builder()

        bg_q = transforms3d.quaternions.axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(90)
        )
        if self.scene_dataset == "ProcTHOR":
            # for some reason the scene needs to rotate around y-axis by 90 degrees for ProcTHOR scenes from hssd dataset
            bg_q = transforms3d.quaternions.qmult(
                bg_q,
                transforms3d.quaternions.axangle2quat(
                    np.array([0, -1, 0]), theta=np.deg2rad(90)
                ),
            )
        bg_pose = sapien.Pose(q=bg_q)
        builder.add_visual_from_file(bg_path, pose=bg_pose)
        builder.add_nonconvex_collision_from_file(bg_path, pose=bg_pose)
        self.bg = builder.build_static(name="scene_background")

        global_id = 0
        for object in tqdm(scene_json["object_instances"][:]):
            model_path = (
                Path(ASSET_DIR)
                / "scene_datasets/ai2thor/ai2thorhab-uncompressed/assets"
                / f"{object['template_name']}.glb"
            )
            actor_id = f"{object['template_name']}_{global_id}"
            global_id += 1
            q = transforms3d.quaternions.axangle2quat(
                np.array([1, 0, 0]), theta=np.deg2rad(90)
            )
            rot_q = [
                object["rotation"][0],
                object["rotation"][1],
                object["rotation"][2],
                object["rotation"][-1],
            ]
            q = transforms3d.quaternions.qmult(q, rot_q)

            builder = scene.create_actor_builder()
            if self._should_be_kinematic(object["template_name"]) or not np.any(
                [name in actor_id.lower() for name in WORKING_OBJS]
            ):
                position = [
                    object["translation"][0],
                    -object["translation"][2],
                    object["translation"][1] + 0,
                ]
                position = [
                    object["translation"][0],
                    -object["translation"][2],
                    object["translation"][1] + 0,
                ]
                pose = sapien.Pose(p=position, q=q)
                builder.add_visual_from_file(str(model_path), pose=pose)
                builder.add_nonconvex_collision_from_file(str(model_path), pose=pose)
                actor = builder.build_static(name=actor_id)
            else:
                position = [
                    object["translation"][0],
                    -object["translation"][2],
                    object["translation"][1] + 0.005,
                ]
                builder.add_visual_from_file(str(model_path))
                builder.add_multiple_convex_collisions_from_file(
                    str(model_path),
                    decomposition='coacd' if self.config.convex_decomposition else 'none',
                )
                actor = builder.build(name=actor_id)
                self._movable_objects[actor.name] = actor
                pose = sapien.Pose(p=position, q=q)
                self.actor_default_poses.append((actor, pose))
            self._scene_objects[actor_id] = actor

        # if self._scene_navigable_positions[scene_idx] is None:
        #     self._scene_navigable_positions[scene_idx] = np.load(
        #         Path(dataset.dataset_path)
        #         / (
        #             Path(scene_cfg.config_file).stem.split(".")[0]
        #             + f".{self.env.robot_uids}.navigable_positions.npy"
        #         )
        #     )

    
    def _get_sapien_entity(self):
        # return self.scene_objecte
        return list(self._movable_objects.values())

    def _should_be_kinematic(self, template_name: str):
        object_config_json = (
            Path(ASSET_DIR)
            / "scene_datasets/ai2thor/ai2thorhab-uncompressed/configs"
            / f"{template_name}.object_config.json"
        )
        with open(object_config_json, "r") as f:
            object_config_json = json.load(f)
        semantic_id = object_config_json["semantic_id"]
        object_category = SEMANTIC_ID_OBJECT_MAPPING[semantic_id]
        return object_category not in MOVEABLE_OBJECT_IDS