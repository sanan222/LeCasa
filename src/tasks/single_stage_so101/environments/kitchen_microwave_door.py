from robocasa.environments.kitchen.kitchen_simple import KitchenSimple
from robocasa.models.fixtures import FixtureType, Microwave


class MicrowaveDoor(KitchenSimple):
    """
    Class encapsulating the atomic microwave door manipulation tasks.

    Args:
        behavior (str): "open" or "close". Used to define the desired
            microwave door manipulation behavior for the task.
    """

    def __init__(self, behavior="open", *args, **kwargs):
        assert behavior in ["open", "close"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the microwave door tasks
        """
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref("microwave", dict(id=FixtureType.MICROWAVE))
        self.counter = self.get_fixture(FixtureType.COUNTER)
        self.init_robot_base_pos = self.microwave

    def _reset_internal(self):
        """
        Reset the environment internal state for the microwave door tasks.
        This includes setting the door state based on the behavior
        """
        if self.behavior == "open":
            self.microwave.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
        elif self.behavior == "close":
            self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        # set the door state then place the objects otherwise objects initialized in opened microwave will fall down before the door is opened
        super()._reset_internal()

    def get_ep_meta(self):
        """
        Get the episode metadata for the microwave door tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"{self.behavior} the microwave door"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the microwave door tasks. This includes the object placement configurations.
        Place one object inside the microwave and 1-4 distractors on the counter.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="door_obj",
                obj_groups="all",
                graspable=True,
                placement=dict(
                    fixture=self.microwave,
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="container",
                ),
            )
        )

        # Add distractors on counter
        for i in range(1, 4):
            cfgs.append(
                dict(
                    name=f"distractor_{i}",
                    obj_groups="all",
                    graspable=True,
                    placement=dict(
                        fixture=self.counter,
                        ref=self.microwave,
                        size=(0.25, 0.30),
                        pos=("ref", 0.0),
                        offset=(0.2 * i, 0.0),
                        ensure_object_boundary_in_range=False,
                        margin=0.02,
                    ),
                )
            )

        return cfgs

    def _check_success(self):
        """
        Check if the microwave door manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        door_state = self.microwave.get_door_state(env=self)

        success = True
        for joint_p in door_state.values():
            if self.behavior == "open":
                if joint_p < 0.90:
                    success = False
                    break
            elif self.behavior == "close":
                if joint_p > 0.05:
                    success = False
                    break

        return success

