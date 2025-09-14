"""
Kitchen environments for banana to bowl pick and place task.
Simple task for placing a banana and bowl on a table for robotic manipulation.

This environment sets up the kitchen with fixtures (sink, counter) and objects (banana, bowl, apple, can).
Objects are initially placed using standard fixture-based placement, then repositioned relative 
to the robot base by the teleop controller after the robot is positioned.

The robot base is positioned relative to the sink, and then objects are placed at precise
locations relative to the robot base for consistent, controllable layouts.
"""

import numpy as np
from tasks.kitchen_simple import KitchenSimple
from robocasa.models.fixtures import FixtureType


class BananaToBowlPnP(KitchenSimple):
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
        Objects will be positioned using standard fixture-based placement initially,
        then repositioned relative to robot base by the teleop controller.
        """
        cfgs = []
        
        # Banana (graspable target object) - initial placement on counter
        cfgs.append(
            dict(
                name="banana",
                obj_groups="banana",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    ref=self.sink,
                    size=(0.25, 0.30),
                    pos=("ref", 0.0),
                    offset=(0.3, 0.0),
                    ensure_object_boundary_in_range=False,
                    margin=0.02,
                ),
            )
        )

        # Bowl (target container) - initial placement on counter
        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl", 
                graspable=False,
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
        
        # Apple - initial placement on counter
        cfgs.append(
            dict(
                name="apple",
                obj_groups="apple", 
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    ref=self.sink,
                    size=(0.25, 0.30),
                    pos=("ref", 0.2),
                    offset=(0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                    margin=0.02,
                ),
            )
        )
        
        # Can - initial placement on counter
        cfgs.append(
            dict(
                name="can_4",
                obj_groups="can", 
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    ref=self.sink,
                    size=(0.25, 0.30),
                    pos=("ref", 0.25),
                    offset=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                    margin=0.00,
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