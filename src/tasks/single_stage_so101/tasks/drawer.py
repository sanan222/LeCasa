import xml.etree.ElementTree as ET

from tasks.single_stage_so101.environments.kitchen_drawer_simple import (
    ManipulateDrawerSimple as _EnvManipulateDrawer,
)
from tasks.teleoperator import SO101TeleoperationController


class DrawerTeleop(SO101TeleoperationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kitchen_env = None

    def initialize_kitchen_environment(self):
        self.kitchen_env = _EnvManipulateDrawer(
            layout_ids=[self.layout_id],
            style_ids=[self.style_id],
            seed=123,
        )
        return self.kitchen_env

    def _position_robot_base(self, kitchen_xml, robot_xml):
        base = robot_xml.root.find(".//body[@name='base']")
        if base is None:
            return base
        
        
        
        # Fallback: try to find drawer and position relative to it
        drawer = None
        drawer_names = [
            # Stack 1 drawers
            "stack_1_main_group_1_main",
            "stack_1_main_group_2_main", 
            "stack_1_main_group_3_main",
            "stack_1_main_group_4_main",
            # Stack 2 drawers
            "stack_2_main_group_1_main",
            "stack_2_main_group_2_main",
            # Stack 3 drawers
            "stack_3_main_group_1_main",
            # Stack 4 drawers
            "stack_4_main_group_1_main",
            "stack_4_main_group_2_main",
            # Fallback names
            "drawer",
            "drawer_main_group_main",
            "drawer_group_main",
            "drawer_main"
        ]
        
        for name in drawer_names:
            drawer = kitchen_xml.root.find(f".//body[@name='{name}']")
            if drawer is not None:
                break
                
        if drawer is None:
            print("Warning: Could not find coffee machine or drawer body in kitchen XML, using default position")
            base.set("pos", "1.0 -0.7 0.0")
            base.set("quat", "1 0 0 0")
            return base
            
        dx, dy, dz = map(float, (drawer.get("pos", "0 0 0").split()))
        base.set("pos", f"{dx:.4f} {dy - 0.75:.4f} {dz + 0.2:.4f}")
        base.set("quat", "1 0 0 1")
        return base

    def _add_overhead_camera(self, kitchen_xml, robot_base_body=None):
        worldbody = kitchen_xml.root.find(".//worldbody")
        if worldbody is None:
            return None
        cam = ET.SubElement(worldbody, "camera")
        cam.set("name", "overhead_camera")
        if robot_base_body is not None and robot_base_body.get("pos"):
            bx, by, bz = map(float, robot_base_body.get("pos").split())
            cam.set("pos", f"{bx:.4f} {by - 0.3:.4f} {bz + 1.05:.4f}")
        else:
            cam.set("pos", "1.0 -0.3 2.0")
        cam.set("quat", "0.707107 0 0 0.707107")
        cam.set("fovy", "60")
        return cam
    
    def _add_base_stand(self, kitchen_xml, robot_base_body):
        worldbody = kitchen_xml.root.find(".//worldbody")
        if worldbody is None:
            return None
        
        # Get robot base position
        if robot_base_body is not None and robot_base_body.get("pos"):
            bx, by, bz = map(float, robot_base_body.get("pos").split())
            # Position the base stand directly under the robot base
            stand_x, stand_y, stand_z = bx, by, bz/2  # 10cm below robot base
        else:
            # Fallback position if robot base position is not available
            stand_x, stand_y, stand_z = 1.0, -0.7, -0.1
            
        base_stand = ET.SubElement(worldbody, "body")
        base_stand.set("name", "base_stand")
        base_stand.set("pos", f"{stand_x:.4f} {stand_y:.4f} {stand_z:.4f}")
        base_stand.set("quat", "1 0 0 0")
        
        # Add geom for the base stand
        geom = ET.SubElement(base_stand, "geom")
        geom.set("name", "base_stand_geom")
        geom.set("size", f"0.2 0.2 {stand_z:.4f}")  # Wider and flatter base
        geom.set("type", "box")
        geom.set("group", "1")
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("friction", "1.0 0.5 0.01")
        geom.set("rgba", "0.4 0.2 0.1 1")  # Brown color
        
        return base_stand
