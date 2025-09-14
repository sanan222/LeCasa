from tasks.kitchen_simple import KitchenSimple
from robocasa.models.fixtures import FixtureType


class ManipulateDrawerSimple(KitchenSimple):
    """
    Simplified drawer manipulation task that avoids robot initialization issues.
    
    Args:
        behavior (str): "open" or "close". Used to define the desired
            drawer manipulation behavior for the task.
        drawer_id (str): The drawer fixture id to manipulate
    """

    def __init__(self, behavior="open", drawer_id=FixtureType.TOP_DRAWER, *args, **kwargs):
        self.drawer_id = drawer_id
        assert behavior in ["open", "close"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the drawer tasks
        """
        super()._setup_kitchen_references()
        self.drawer = self.register_fixture_ref("drawer", dict(id=self.drawer_id))
        self.init_robot_base_pos = self.drawer

    def _reset_internal(self):
        """
        Reset the environment internal state for the drawer tasks.
        This includes setting the drawer state based on the behavior
        """
        if self.behavior == "open":
            self.drawer.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
        elif self.behavior == "close":
            self.drawer.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        # set the door state then place the objects otherwise objects initialized in opened drawer will fall down before the drawer is opened
        super()._reset_internal()

    def get_ep_meta(self):
        """
        Get the episode metadata for the drawer tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"{self.behavior} the drawer"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the drawer tasks. This includes the object placement configurations.
        Place one object inside the drawer.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="drawer_obj",
                obj_groups="all",
                graspable=True,
                placement=dict(
                    fixture=self.drawer,
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="container",
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the drawer manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        door_state = self.drawer.get_door_state(env=self)

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
