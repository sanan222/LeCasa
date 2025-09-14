"""
Wrapper task class for orange to plate pick-and-place using the RoboCasa SO101-compatible
implementation, exposed under our local `tasks` namespace for the CLI.
"""
from typing import Optional
from tasks.single_stage_so101.environments.kitchen_pantosink import (
    PanToSinkPnP as _EnvPanToSinkPnP,
)
from tasks.teleoperator import SO101TeleoperationController
import xml.etree.ElementTree as ET


class PanToSinkTeleop(SO101TeleoperationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kitchen_env = None

    def initialize_kitchen_environment(self):
        self.kitchen_env = _EnvPanToSinkPnP(
            layout_ids=[self.layout_id],
            style_ids=[self.style_id],
            seed=123,
        )
        return self.kitchen_env

    def _position_robot_base(self, kitchen_xml, robot_xml) -> Optional[ET.Element]:
        """
        Position the robot base body at a suitable counter location.

        Expects a MujocoXML-like object with a .root Element (from mujoco_utils.mujoco_xml.MujocoXML).
        Returns the base body element if found, else None.
        """
        if robot_xml is None or getattr(robot_xml, "root", None) is None:
            return None
        
        # Try to find sink using multiple possible names
        sink_body = None
        sink_names = [
            "sink_main_group_main",
            "sink",
            "sink_group_main",
            "sink_main"
        ]
        
        for name in sink_names:
            sink_body = kitchen_xml.root.find(f".//body[@name='{name}']")
            if sink_body is not None:
                break
        
        if sink_body is not None:
            sink_pos = sink_body.get("pos")
            print(f"Sink pos: {sink_pos}")
        else:
            # Fallback: try to find any counter or use a default position
            counter_body = kitchen_xml.root.find(".//body[@name='counter_main_main_group_main']")
            if counter_body is not None:
                counter_pos = counter_body.get("pos")
                print(f"Counter pos: {counter_pos}")
                sink_pos = counter_pos
            else:
                print("Warning: Could not find sink or counter body in kitchen XML, using default position")
                sink_pos = "1.25 -0.3 0.96"
            
        base_body = robot_xml.root.find(".//body[@name='base']")
        if base_body is not None:
            sink_pos = sink_pos.split()
            print(sink_pos)
            base_body.set("pos", f"{float(sink_pos[0]) + 0.4} {(float(sink_pos[1]) - 0.34)} {float(sink_pos[2]) - 0.02}")
            base_body.set("quat", "1 0 0 1")
            print(f"Base pos: {base_body.get('pos')}")
            return base_body

    def _add_overhead_camera(self, kitchen_xml, robot_base_body):
        """
        Add an overhead camera to the kitchen scene. If robot base position is provided,
        place the camera relative to it, otherwise use a sensible default.

        Expects a MujocoXML-like object with a .root Element.
        Returns the camera element if created, else None.
        """
        if kitchen_xml is None or getattr(kitchen_xml, "root", None) is None:
            return None

        worldbody = kitchen_xml.root.find(".//worldbody")

        # Arrange camera position based on robot base position
        pos_attr = robot_base_body.get("pos")
        bx, by, bz = map(float, pos_attr.split())
        cam_pos = (bx, by + 0.25, bz + 0.48)
        cam_quat = (0.0, 0.0, 0.0, 1.0)
                

        overhead_camera = ET.SubElement(worldbody, "camera")
        overhead_camera.set("name", "overhead_camera")
        overhead_camera.set("pos", f"{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}")
        overhead_camera.set("quat", f"{cam_quat[0]} {cam_quat[1]} {cam_quat[2]} {cam_quat[3]}")
        overhead_camera.set("fovy", "60")
        return overhead_camera
    
    
    
    def _fix_object_location(self, kitchen_xml, object_name, new_location):
        
        """
        Change the location of an object in the kitchen scene relative to the robot base.
        """
        
        robot_base_body = kitchen_xml.root.find(".//body[@name='base']")
        if robot_base_body is None:
            print(f"Warning: robot base body not found")
            return
        robot_base_pos = robot_base_body.get("pos")
        robot_base_pos = robot_base_pos.split()
        
        new_location = f"{float(robot_base_pos[0]) + float(new_location[0])} {float(robot_base_pos[1]) + float(new_location[1])} {float(robot_base_pos[2]) + float(new_location[2])}"
        
        
        
        object_body = kitchen_xml.root.find(f".//body[@name='{object_name}']")
        if object_body is None:
            print(f"Warning: {object_name} body not found")
            return
        object_body.set("pos", new_location)
        # object_body.set("quat", "1 0 0 1")
        return object_body
    
    


