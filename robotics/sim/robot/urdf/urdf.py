import os
import copy
import xml.etree.ElementTree as ET
import logging
from xml.etree.ElementTree import Element
from typing import Optional, Sequence, TypeVar
from queue import Queue

import sapien
import tempfile

from robotics import Pose


def get_str(elem: Element, attr: str) -> str:
    a = elem.get(attr)
    assert a is not None, f"attribute {attr} not found"
    return a


T = TypeVar('T', bound='URDFElement')
class URDFElement:
    type: str
    def __init__(self, elem: Element) -> None:
        self.elem = elem

    def add_content(self, content):
        self.elem.extend(content)

    def content(self):
        for i in self.elem:
            yield i

    def replace_package(self: T, package_dir: str, new_dir: str) -> T:
        return self

    @property
    def name(self) -> str:
        return get_str(self.elem, 'name')
        

def prune_package(path: str):
    if path.startswith('package://'):
        path = path.replace('package://', '')
    return path

class Link(URDFElement):
    type = 'link'


class Joint(URDFElement):
    type = 'joint'

    @property
    def parent(self):
        parent = self.elem.find('parent')
        assert parent is not None
        return get_str(parent, 'link')

    @property
    def child(self):
        child = self.elem.find('child')
        assert child is not None
        return get_str(child, 'link')

    def set_parent(self, parent):
        _parent_elem = self.elem.find('parent')
        assert _parent_elem is not None
        _parent_elem.set('link', parent)

    @classmethod
    def new(cls, name, parent: str, child: str, type: str = 'fixed', rpy: Optional[Sequence[float]]=None, xyz: Optional[Sequence[float]]=None):
        joint = ET.Element('joint')
        joint.set('type', type)
        joint.set('name', name)
        _parent = ET.SubElement(joint, 'parent')
        _parent.set('link', parent)
        _child = ET.SubElement(joint, 'child')
        _child.set('link', child)

        _origin = ET.SubElement(joint, 'origin')
        xyz = xyz or [0, 0, 0]
        _origin.set('xyz', ' '.join(map(str, xyz)))
        rpy = rpy or [0, 0, 0]
        _origin.set('rpy', ' '.join(map(str, rpy)))
        return cls(joint)

        
        


class URDFTool:
    @classmethod
    def from_path(cls, base):
        tree = ET.parse(base)
        root: Element = tree.getroot()
        assert root.tag == 'robot'
        return cls(root, os.path.dirname(base))

    def __init__(self, root: Element, package_dir: str) -> None:
        self.root = root
        all_links = [Link(i) for i in root.findall('link')]
        all_joints = [Joint(j) for j in root.findall('joint')]
        self.all_links = {
            i.name: i for i in all_links
        }
        self.all_joints = {
            i.name: i for i in all_joints
        }
        self.package_dir = package_dir
        self.others = [i for i in self.root if i.tag not in ['link', 'joint']]


    def remove(self, name, dtype=None):
        if dtype != 'joint':
            if name in self.all_links:
                self.root.remove(self.all_links[name].elem)
                self.all_links.pop(name)

        if dtype != 'link':
            if name in self.all_joints:
                self.root.remove(self.all_joints[name].elem)
                self.all_joints.pop(name)

    def get(self, name: str) -> URDFElement:
        if name in self.all_links:
            return self.all_links[name]
        if name in self.all_joints:
            return self.all_joints[name]
        raise KeyError(f"element {name} not found")

    def bfs(self, origin: str, callback=None) -> Sequence[URDFElement]:
        # find all links that are connected to the origin
        queue: Queue[Link] = Queue()
        queue.put(self.all_links[origin])

        #results = set() 
        results = []
        #results.add(origin)
        results.append(self.all_links[origin])

        while not queue.empty():
            link = queue.get()
            # print('bfs', link.name)
            for joint in self.all_joints.values():
                if joint.parent == link.name:
                    other = joint.child
                elif joint.child == link.name:
                    other = joint.parent
                else:
                    continue
                if other not in self.all_links:
                    continue

                other_link = self.all_links[other]
                if other_link not in results:
                    queue.put(other_link)

                    results.append(joint)
                    results.append(other_link)

                    if callback is not None:
                        callback(locals())


        return results


    def find_root(self, origin: Optional[str], index: int=0) -> Link:
        # remove all links and joints that are not connected to the origin or the indexed elements
        if origin is not None:
            #origin = self.root.find('link').get('name')
            link = self.all_links[origin]
        else:
            if index not in [0, -1]:
                logging.warning(f'index {index} is not 0 or -1, which is not recommended')
            links = [i for i in self.all_links.values()]
            assert len(links) > 0
            link = links[index]


        num_parents = {}
        def count_parents(locals_):
            link = locals_['link']
            joint = locals_['joint']
            other = locals_['other_link']
            is_parent = link.name == joint.parent
            if not is_parent:
                num_parents[link.name] = num_parents.get(link.name, 0) + 1
            else:
                num_parents[other.name] = num_parents.get(other.name, 0) + 1
            

        results = self.bfs(link.name, callback=count_parents)

        root = None
        for i in results:
            if i.type == 'link' and i.name not in num_parents:
                assert root is None, f"multiple roots found: {root.name}, {i.name}"
                root = i
            
        assert root is not None, f"no root found: circular dependency detected"
        assert isinstance(root, Link)
        return root
    
            
    def prune_from(self, origin: Optional[str]=None, index: int=0):
        root = self.find_root(origin, index)
        # print('prune from', root.name)
        results = self.bfs(root.name)

        new_root = ET.Element('robot')
        for i in results:
            # print(i.name, type(i))
            new_root.append(i.elem)
        for i in self.others:
            new_root.append(i)
        new_root.set('name', root.name)
        return URDFTool(copy.deepcopy(new_root), package_dir=self.package_dir)


    def export(self, path, robot_name: Optional[str]=None, absolute=False):
        root = copy.deepcopy(self.root)
        new_dir = os.path.dirname(path)
        print(self.package_dir, new_dir)

        for elem in root.iter():
            if 'filename' in elem.attrib:
                if not absolute:
                    elem.attrib['filename'] = os.path.relpath(os.path.join(self.package_dir, prune_package(elem.attrib['filename'])), new_dir)
                else:
                    elem.attrib['filename'] = os.path.abspath(os.path.join(self.package_dir, prune_package(elem.attrib['filename'])))

        if robot_name is not None:
            root.set('name', robot_name)
        tree = ET.ElementTree(root)
        tree.write(path, encoding='utf-8', xml_declaration=True)
        return tree

        
    def load(self, scene: sapien.Scene, fix_root_link: bool=True, filename: Optional[str]=None, pose: Optional[sapien.Pose]=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            if filename is None:
                import hashlib
                filename = hashlib.md5(ET.tostring(self.root)).hexdigest() + '.urdf'
                tmp_path = os.path.join(tmpdir, f'{filename}')
            else:
                tmp_path = filename
            self.export(tmp_path)

            loader = scene.create_urdf_loader()
            loader.fix_root_link = fix_root_link
            robot = loader.load(tmp_path)

            if pose is not None:
                robot.set_root_pose(pose)

            return robot, tmp_path

            
    def add(
        self, 
        other: "URDFTool",
        pose: Pose,
        base_name: str,
        joint_name: str,
        joint_type: str='fixed',
    ) -> "URDFTool":
        # print("+++++")

        assert joint_type == 'fixed', f"joint type {joint_type} not supported"
        other_root = other.find_root(None, 0)
        other = other.prune_from(other_root.name)
        import transforms3d
        
        child_name = other_root.name

        rpy = transforms3d.euler.quat2euler(pose.q)
        p = list(pose.p)

        joint = Joint.new(joint_name, base_name, child_name, joint_type, rpy, p)

        import copy
        new_root = copy.deepcopy(self.root)
        new_root.append(joint.elem)

        for j in other.root.iter():
            if 'filename' in j.attrib:
                j.attrib['filename'] = os.path.relpath(os.path.join(other.package_dir, prune_package(j.attrib['filename'])), self.package_dir)

        materials = set()
        for i in self.root:
            if i.tag == 'material':
                materials.add(i.get('name'))

        for i in other.root:
            if i.tag == 'material' and i.get('name') in materials:
                continue
            new_root.append(i)

        return URDFTool(new_root, self.package_dir)