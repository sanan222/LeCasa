"""
Kitchen environments for orange to plate pick and place task.
Simple task for placing an orange and plate on a table for robotic manipulation.
"""

import numpy as np
from robocasa.environments.kitchen.kitchen_simple import KitchenSimple
from robocasa.models.fixtures import FixtureType


class OrangeToPlatePnP(KitchenSimple):
    """
    KitchenSimple-compatible orange to plate pick and place task.
    
    Places an orange (graspable) and a plate on a counter surface near the sink.
    The robot should pick up the orange and place it on the plate.
    """

    def __init__(self, *args, **kwargs):
        # Set a fixed seed for consistent object placement if not provided
        if 'seed' not in kwargs or kwargs['seed'] is None:
            kwargs['seed'] = 123  # Fixed seed for OrangeToPlatePnP task
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the banana to bowl pick and place task:
        The sink as reference point and the counter to place objects on
        """
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_pos = self.sink

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the orange to plate pick and place task.
        Places an orange and plate on a counter next to the sink.
        """
        cfgs = []
        
        # Orange (graspable target object) - placed to the left of the plate
        cfgs.append(
            dict(
                name="orange",
                obj_groups="orange",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    ref=self.sink,
                    size=(0.25, 0.30),
                    pos=("ref", 0.0),  # Further left of bowl position
                    offset=(0.4, 0.0),
                    ensure_object_boundary_in_range=False,
                    margin=0.02,
                ),
            )
        )

        # Bowl (target container) - moved more to the left
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate", 
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    ref=self.sink,
                    size=(0.25, 0.30),
                    pos=("ref", 0.0),  # Moved left from 0.8 to 0.6
                    offset=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                    margin=0.02,
                ),
            )
        )
        
        # cfgs.append(
        #     dict(
        #         name="can_4",
        #         obj_groups="can",
        #         graspable=False,
        #         placement=dict(
        #             fixture=self.counter,
        #             ref=self.sink,
        #             size=(0.25, 0.30),
        #             pos=("ref", 0.0),
        #             offset=(0.0, 0.0),
        #             ensure_object_boundary_in_range=False,
        #             margin=0.02,
        #         ),
        #     )
        # )
        
        # cfgs.append(
        #     dict(
        #         name="apple",
        #         obj_groups="apple",
        #         graspable=False,
        #         placement=dict(
        #             fixture=self.counter,
        #             ref=self.sink,
        #             size=(0.25, 0.30),
        #             pos=("ref", 0.0),
        #             offset=(0.0, 0.0),
        #             ensure_object_boundary_in_range=False,
        #             margin=0.02,
        #         ),
        #     )
        # )
        
        # cfgs.append(
        #     dict(
        #         name="banana",
        #         obj_groups="banana",
        #         graspable=False,
        #         placement=dict(
        #             fixture=self.counter,
        #             ref=self.sink,
        #             size=(0.25, 0.30),
        #             pos=("ref", 0.0),
        #             offset=(0.0, 0.0),
        #             ensure_object_boundary_in_range=False,
        #             margin=0.02,
        #         ),
        #     )
        # )
        
        # cfgs.append(
        #     dict(
        #         name="mango",
        #         obj_groups="mango",
        #         graspable=False,
        #         placement=dict(
        #             fixture=self.counter,
        #             ref=self.sink,
        #             size=(0.25, 0.30),
        #             pos=("ref", 0.0),
        #             offset=(0.0, 0.0),
        #             ensure_object_boundary_in_range=False,
        #             margin=0.02,
        #         ),
        #     )
        # )
        
        # cfgs.append(
        #     dict(
        #         name="peach",
        #         obj_groups="peach",
        #         graspable=False,
        #         placement=dict(
        #             fixture=self.counter,
        #             ref=self.sink,
        #             size=(0.25, 0.30),
        #             pos=("ref", 0.0),
        #             offset=(0.0, 0.0),
        #             ensure_object_boundary_in_range=False,
        #             margin=0.02,
        #         ),
        #     )
        # )

        return cfgs

    def _check_success(self):
        """
        Check if the orange to plate pick and place task is successful.
        Checks if the orange is on the plate and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        # Check if orange is on plate (using object-in-receptacle check)
        try:
            import robocasa.utils.object_utils as OU
            orange_on_plate = OU.check_obj_in_receptacle(self, "orange", "plate")
            gripper_obj_far = OU.gripper_obj_far(self)
            return orange_on_plate and gripper_obj_far
        except:
            # Fallback: simple position-based check
            orange_obj = self.objects.get("orange")
            plate_obj = self.objects.get("plate")
            if orange_obj is None or plate_obj is None:
                return False
            
            # Check if orange is close to plate position
            orange_pos = orange_obj.get_position()
            plate_pos = plate_obj.get_position()
            distance = np.linalg.norm(orange_pos[:2] - plate_pos[:2])  # 2D distance
            return distance < 0.1  # Within 10cm

    def get_ep_meta(self):
        """
        Get the episode metadata for the orange to plate pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick the orange from the counter and place it on the plate"
        return ep_meta 