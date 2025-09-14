"""
Kitchen environments for orange to bowl pick and place task.
Simple task for placing an orange and bowl on a table for robotic manipulation.
"""

import numpy as np
from tasks.kitchen_simple import KitchenSimple
from robocasa.models.fixtures import FixtureType


class PanToSinkPnP(KitchenSimple):
    """
    KitchenSimple-compatible orange to bowl pick and place task.
    
    Places an orange (graspable) and a bowl on a counter surface near the sink.
    The robot should pick up the orange and place it in the bowl.
    """

    def __init__(self, *args, **kwargs):
        # Set a fixed seed for consistent object placement if not provided
        if 'seed' not in kwargs or kwargs['seed'] is None:
            kwargs['seed'] = 123  # Fixed seed for OrangeToBowlPnP task
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the orange to bowl pick and place task:
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
        Get the object configurations for the orange to bowl pick and place task.
        Places an orange and bowl on a counter next to the sink.
        """
        cfgs = []
        
        # Orange (graspable target object) - placed closer to the sink for better accessibility
        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    ref=self.sink,  # use sink as reference fixture
                    size=(0.25, 0.30),
                    # align x with reference, place closer to sink
                    pos=("ref", 0.0),
                    # smaller offset to keep orange closer to bowl
                    offset=(0.2, 0.0),
                    ensure_object_boundary_in_range=False,
                    margin=0.02,
                ),
            )
        )
        
        cfgs.append(
            dict(
                name="sponge",
                obj_groups="sponge",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    ref=self.sink,
                    size=(0.25, 0.30),
                    pos=("ref", 0.0),
                    offset=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                    margin=0.02,
                ),
            )
            
        )

        
        return cfgs

    def _check_success(self):
        """
        Check if the orange to bowl pick and place task is successful.
        Checks if the orange is in the bowl and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        # Check if orange is in bowl (using object-in-receptacle check)
        try:
            import robocasa.utils.object_utils as OU
            pan_in_sponge = OU.check_obj_in_receptacle(self, "pan", "sponge")
            gripper_obj_far = OU.gripper_obj_far(self)
            return pan_in_sponge and gripper_obj_far
        except:
            # Fallback: simple position-based check
            pan_obj = self.objects.get("pan")
            sponge_obj = self.objects.get("sponge")
            if pan_obj is None or sponge_obj is None:
                return False
            
            # Check if pan is close to sponge position
            pan_pos = pan_obj.get_position()
            sponge_pos = sponge_obj.get_position()
            distance = np.linalg.norm(pan_pos[:2] - sponge_pos[:2])  # 2D distance
            return distance < 0.1  # Within 10cm

    def get_ep_meta(self):
        """
        Get the episode metadata for the pan to sponge pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick the pan from the counter and place it in the sponge"
        return ep_meta
