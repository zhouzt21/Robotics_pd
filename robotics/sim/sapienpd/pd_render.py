import sapien
from sapien.render import RenderCudaMeshComponent
import warp as wp
import numpy as np

from .pd_component import *
from .pd_system import PDSystem
from .utils.logger import sapienpd_logger as logger


def scene_update_render(scene: sapien.Scene, set_body_pose=False):
    sys: PDSystem = scene.get_system("pd")

    for entity in scene.get_entities():
        for component in entity.get_components():
            if isinstance(component, PDClothComponent):
                render_component: RenderCudaMeshComponent = entity.find_component_by_type(
                    RenderCudaMeshComponent
                )
                if render_component is None:
                    logger.warning(
                        f"Entity {entity.name} has a PDClothComponent but no RenderCudaMeshComponent"
                    )
                    continue
                component.update_render(render_component)

            if isinstance(component, PDBodyComponent):
                if set_body_pose:
                    component.update_entity_pose()

    scene.update_render()
