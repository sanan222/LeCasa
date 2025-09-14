"""
Kitchen environments for banana to bowl pick and place task.
Simple task for placing a banana and bowl on a table for robotic manipulation.
"""

import numpy as np
from tasks.kitchen_simple import KitchenSimple
from robocasa.models.fixtures import FixtureType


class FruitBowlPnP(KitchenSimple):
    """
    KitchenSimple-compatible banana to bowl pick and place task.
    
    Places a banana (graspable) and a bowl on a counter surface near the sink.
    The robot should pick up the banana and place it in the bowl.
    """

    def __init__(self, *args, **kwargs):
        # Set a fixed seed for consistent object placement if not provided
        if 'seed' not in kwargs or kwargs['seed'] is None:
            kwargs['seed'] = 123  # Fixed seed for BananaToBowlPnP task
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
        Get the object configurations for the banana to bowl pick and place task.
        Places a banana and bowl on a counter next to the sink.
        """
        cfgs = []
        
        # Banana (graspable target object) - placed on the right side of sink-referenced counter
        cfgs.append(
            dict(
                name="banana",
                obj_groups="banana",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    ref=self.sink,  # use sink as reference fixture
                    size=(0.25, 0.30),
                    # align x with reference, push to right side (positive y)
                    pos=("ref", 0.0),
                    # small offset to separate from bowl
                    offset=(0.4, 0.0),
                    ensure_object_boundary_in_range=False,
                    margin=0.02,
                ),
            )
        )

        # Bowl (target container) - offset along +x relative to the sink-referenced counter
        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl", 
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    ref=self.sink,  # use sink as reference fixture
                    size=(0.25, 0.30),
                    # align x with reference; move along +x using offset
                    pos=("ref", 0.0),
                    offset=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                    margin=0.02,
                ),
            )
        )
        
        # Orange (graspable target object) - placed on the right side of sink-referenced counter
        cfgs.append(
            dict(
                name="peach",
                obj_groups="peach",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    ref=self.sink,  # use sink as reference fixture
                    size=(0.25, 0.30),
                    # align x with reference, push to right side (positive y)
                    pos=("ref", 0.0),
                    # small offset to separate from bowl
                    offset=(0.4, -0.1),
                    ensure_object_boundary_in_range=False,
                    margin=0.02,
                ),
            )
        )

        # Orange (target container) - offset along +x relative to the sink-referenced counter
        cfgs.append(
            dict(
                name="orange",
                obj_groups="orange", 
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    ref=self.sink,  # use sink as reference fixture
                    size=(0.25, 0.30),
                    # align x with reference; move along +x using offset
                    pos=("ref", 0.0),
                    offset=(0.5, 0.0),
                    ensure_object_boundary_in_range=False,
                    margin=0.02,
                ),
            )
        )


        return cfgs

    def _check_success(self):
        """
        Check if the banana to bowl pick and place task is successful.
        Checks if the banana is in the bowl and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        # Check if banana is in bowl (using object-in-receptacle check)
        try:
            import robocasa.utils.object_utils as OU
            banana_in_bowl = OU.check_obj_in_receptacle(self, "banana", "bowl")
            gripper_obj_far = OU.gripper_obj_far(self)
            return banana_in_bowl and gripper_obj_far
        except:
            # Fallback: simple position-based check
            banana_obj = self.objects.get("banana")
            bowl_obj = self.objects.get("bowl")
            if banana_obj is None or bowl_obj is None:
                return False
            
            # Check if banana is close to bowl position
            banana_pos = banana_obj.get_position()
            bowl_pos = bowl_obj.get_position()
            distance = np.linalg.norm(banana_pos[:2] - bowl_pos[:2])  # 2D distance
            return distance < 0.1  # Within 10cm

    def get_ep_meta(self):
        """
        Get the episode metadata for the banana to bowl pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick the banana from the counter and place it in the bowl"
        return ep_meta 