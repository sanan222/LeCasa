import numpy as np
from copy import deepcopy

import robocasa.macros as macros
import robocasa.utils.object_utils as OU
import robocasa.models.scenes.scene_registry as SceneRegistry
from robocasa.models.scenes import KitchenArena
from robocasa.models.fixtures import *
from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object
from robocasa.models.objects.objects import MJCFObject
from robocasa.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robocasa.utils.errors import RandomizationError
from robocasa.utils import transform_utils_local as T
from robocasa.utils.mjcf_utils_local import array_to_string
from robocasa.utils.texture_swap import (
    get_random_textures,
    replace_cab_textures,
    replace_counter_top_texture,
    replace_floor_texture,
    replace_wall_texture,
)

# Create a ManipulationTask-like structure with arena and fixtures
from robosuite.models.tasks import ManipulationTask


class KitchenSimple:
    """
    Simplified Kitchen environment that only handles arena XML and fixtures.
    No robot integration, no object creation, no camera observations.

    Args:
        layout_ids ((list of) LayoutType or int): layout id(s) to use for the kitchen. 
            -1 and None specify all layouts
            -2 specifies layouts not involving islands/wall stacks, 
            -3 specifies layouts involving islands/wall stacks,
            -4 specifies layouts with dining areas.

        style_ids ((list of) StyleType or int): style id(s) to use for the kitchen. 
            -1 and None specify all styles.

        generative_textures (str): if set to "100p", will use AI generated textures

        seed (int): environment seed. Default is None, where environment is unseeded, ie. random
    """

    EXCLUDE_LAYOUTS = []

    def __init__(
        self,
        layout_ids=None,
        style_ids=None,
        seed=None,
        obj_registries=("objaverse",),
        **kwargs  # Accept any additional keyword arguments and ignore them
    ):
        # Set up random number generator
        self.rng = np.random.RandomState(seed)

        # Handle layout and style IDs
        layout_ids = SceneRegistry.unpack_layout_ids(layout_ids or -1)
        style_ids = SceneRegistry.unpack_style_ids(style_ids or -1)
        self.layout_and_style_ids = [(l, s) for l in layout_ids for s in style_ids]

        # Remove excluded layouts
        self.layout_and_style_ids = [
            (int(l), int(s))
            for (l, s) in self.layout_and_style_ids
            if l not in self.EXCLUDE_LAYOUTS
        ]

        # Object-related settings
        self.obj_registries = obj_registries
        self.obj_instance_split = kwargs.get('obj_instance_split', None)
        self.force_style_textures = kwargs.get('force_style_textures', None)
        self.generative_textures = kwargs.get('generative_textures', None)

        # Initialize episode metadata
        self._ep_meta = {}
        self._curr_gen_fixtures = None

        # Load the kitchen model
        self._load_model()

    def _load_model(self):
        """
        Loads the kitchen arena and fixtures.
        """
        # Determine sample layout and style
        if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:
            self.layout_id = self._ep_meta["layout_id"]
            self.style_id = self._ep_meta["style_id"]
        else:
            # Choose a random index from the layout_and_style_ids list
            random_index = self.rng.choice(len(self.layout_and_style_ids))
            layout_id, style_id = self.layout_and_style_ids[random_index]
            self.layout_id = int(layout_id)
            self.style_id = int(style_id)

        # Setup scene
        self.mujoco_arena = KitchenArena(
            layout_id=self.layout_id,
            style_id=self.style_id,
            rng=self.rng,
        )
        
        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # Setup fixtures
        self.fixture_cfgs = self.mujoco_arena.get_fixture_cfgs()
        self.fixtures = {cfg["name"]: cfg["model"] for cfg in self.fixture_cfgs}

        # Setup fixture locations
        fxtr_placement_initializer = self._get_placement_initializer(
            self.fixture_cfgs, z_offset=0.0
        )
        
        # Try to place fixtures
        for i in range(10):
            try:
                self.fxtr_placements = fxtr_placement_initializer.sample()
                break
            except RandomizationError:
                if i == 9:  # Last attempt failed
                    self._load_model()
                    return
                continue
        
        # Apply fixture positions
        for obj_pos, obj_quat, obj in self.fxtr_placements.values():
            obj.set_pos(obj_pos)
            obj.set_euler(T.mat2euler(T.quat2mat(T.convert_quat(obj_quat, "xyzw"))))

        # Setup internal references related to fixtures
        self._setup_kitchen_references()

        # Create and place objects
        self._create_objects()

        # Setup object locations
        self.placement_initializer = self._get_placement_initializer(self.object_cfgs)
        try:
            self.object_placements = self.placement_initializer.sample(
                placed_objects=self.fxtr_placements
            )
        except RandomizationError:
            self._load_model()
            return

    def _setup_kitchen_references(self):
        """
        Setup fixtures (and their references). This function is called within load_model function for kitchens
        """
        serialized_refs = self._ep_meta.get("fixture_refs", {})
        # Unserialize refs
        self.fixture_refs = {
            k: self.get_fixture(v) for (k, v) in serialized_refs.items()
        }

    def _get_placement_initializer(self, cfg_list, z_offset=0.01):
        """
        Creates a placement initializer for the objects/fixtures based on the specifications in the configurations list

        Args:
            cfg_list (list): list of object/fixture configurations
            z_offset (float): offset in z direction

        Returns:
            SequentialCompositeSampler: placement initializer
        """
        placement_initializer = SequentialCompositeSampler(
            name="SceneSampler", rng=self.rng
        )

        for (obj_i, cfg) in enumerate(cfg_list):
            # Determine which object is being placed
            if cfg["type"] == "fixture":
                mj_obj = self.fixtures[cfg["name"]]
            elif cfg["type"] == "object":
                mj_obj = self.objects[cfg["name"]]
            else:
                print(f"Warning: Unknown object type '{cfg.get('type', 'None')}' for {cfg.get('name', 'unnamed')}")
                continue

            placement = cfg.get("placement", None)
            if placement is None:
                continue
            fixture_id = placement.get("fixture", None)
            if fixture_id is not None:
                # Get fixture to place object on
                fixture = self.get_fixture(
                    id=fixture_id,
                    ref=placement.get("ref", None),
                )

                # Calculate the total available space where object could be placed
                sample_region_kwargs = placement.get("sample_region_kwargs", {})
                reset_region = fixture.sample_reset_region(
                    env=self, **sample_region_kwargs
                )
                outer_size = reset_region["size"]
                margin = placement.get("margin", 0.04)
                outer_size = (outer_size[0] - margin, outer_size[1] - margin)

                # Calculate the size of the inner region where object will actually be placed
                target_size = placement.get("size", None)
                if target_size is not None:
                    target_size = deepcopy(list(target_size))
                    for size_dim in [0, 1]:
                        if target_size[size_dim] == "obj":
                            target_size[size_dim] = mj_obj.size[size_dim] + 0.005
                        if target_size[size_dim] == "obj.x":
                            target_size[size_dim] = mj_obj.size[0] + 0.005
                        if target_size[size_dim] == "obj.y":
                            target_size[size_dim] = mj_obj.size[1] + 0.005
                    inner_size = np.min((outer_size, target_size), axis=0)
                else:
                    inner_size = outer_size

                inner_xpos, inner_ypos = placement.get("pos", (None, None))
                offset = placement.get("offset", (0.0, 0.0))

                # Center inner region within outer region
                if inner_xpos == "ref":
                    # Compute optimal placement of inner region to match up with the reference fixture
                    x_halfsize = outer_size[0] / 2 - inner_size[0] / 2
                    if x_halfsize == 0.0:
                        inner_xpos = 0.0
                    else:
                        # Robustly resolve reference fixture
                        ref_fixt_id = None
                        if (
                            isinstance(placement.get("sample_region_kwargs"), dict)
                            and "ref" in placement["sample_region_kwargs"]
                        ):
                            ref_fixt_id = placement["sample_region_kwargs"]["ref"]
                        elif "ref" in placement:
                            ref_fixt_id = placement["ref"]

                        if ref_fixt_id is None:
                            # Fallback: center align if no reference provided
                            inner_xpos = 0.0
                        else:
                            ref_fixture = self.get_fixture(ref_fixt_id)
                            fixture_to_ref = OU.get_rel_transform(fixture, ref_fixture)[0]
                            outer_to_ref = fixture_to_ref - reset_region["offset"]
                            inner_xpos = outer_to_ref[0] / x_halfsize
                            inner_xpos = np.clip(inner_xpos, a_min=-1.0, a_max=1.0)
                elif inner_xpos is None:
                    inner_xpos = 0.0

                if inner_ypos is None:
                    inner_ypos = 0.0
                # Offset for inner region
                intra_offset = (
                    (outer_size[0] / 2 - inner_size[0] / 2) * inner_xpos + offset[0],
                    (outer_size[1] / 2 - inner_size[1] / 2) * inner_ypos + offset[1],
                )

                # Center surface point of entire region
                ref_pos = fixture.pos + [0, 0, reset_region["offset"][2]]
                ref_rot = fixture.rot

                # x, y, and rotational ranges for randomization
                x_range = (
                    np.array([-inner_size[0] / 2, inner_size[0] / 2])
                    + reset_region["offset"][0]
                    + intra_offset[0]
                )
                y_range = (
                    np.array([-inner_size[1] / 2, inner_size[1] / 2])
                    + reset_region["offset"][1]
                    + intra_offset[1]
                )
                rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))
            else:
                target_size = placement.get("size", None)
                x_range = np.array([-target_size[0] / 2, target_size[0] / 2])
                y_range = np.array([-target_size[1] / 2, target_size[1] / 2])
                rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))
                ref_pos = [0, 0, 0]
                ref_rot = 0.0

            if macros.SHOW_SITES is True:
                """
                Show outer reset region
                """
                pos_to_vis = deepcopy(ref_pos)
                pos_to_vis[:2] += T.rotate_2d_point(
                    [reset_region["offset"][0], reset_region["offset"][1]], rot=ref_rot
                )
                size_to_vis = np.concatenate(
                    [
                        np.abs(
                            T.rotate_2d_point(
                                [outer_size[0] / 2, outer_size[1] / 2], rot=ref_rot
                            )
                        ),
                        [0.001],
                    ]
                )
                site_str = """<site type="box" rgba="0 0 1 0.4" size="{size}" pos="{pos}" name="reset_region_outer_{postfix}"/>""".format(
                    pos=array_to_string(pos_to_vis),
                    size=array_to_string(size_to_vis),
                    postfix=str(obj_i),
                )
                site_tree = ET.fromstring(site_str)
                self.mujoco_arena.worldbody.append(site_tree)

                """
                Show inner reset region
                """
                pos_to_vis = deepcopy(ref_pos)
                pos_to_vis[:2] += T.rotate_2d_point(
                    [np.mean(x_range), np.mean(y_range)], rot=ref_rot
                )
                size_to_vis = np.concatenate(
                    [
                        np.abs(
                            T.rotate_2d_point(
                                [
                                    (x_range[1] - x_range[0]) / 2,
                                    (y_range[1] - y_range[0]) / 2,
                                ],
                                rot=ref_rot,
                            )
                        ),
                        [0.002],
                    ]
                )
                site_str = """<site type="box" rgba="1 0 0 0.4" size="{size}" pos="{pos}" name="reset_region_inner_{postfix}"/>""".format(
                    pos=array_to_string(pos_to_vis),
                    size=array_to_string(size_to_vis),
                    postfix=str(obj_i),
                )
                site_tree = ET.fromstring(site_str)
                self.mujoco_arena.worldbody.append(site_tree)

            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="{}_Sampler".format(cfg["name"]),
                    mujoco_objects=mj_obj,
                    x_range=x_range,
                    y_range=y_range,
                    rotation=rotation,
                    ensure_object_boundary_in_range=placement.get(
                        "ensure_object_boundary_in_range", True
                    ),
                    ensure_valid_placement=placement.get(
                        "ensure_valid_placement", True
                    ),
                    reference_pos=ref_pos,
                    reference_rot=ref_rot,
                    z_offset=z_offset,
                    rng=self.rng,
                    rotation_axis=placement.get("rotation_axis", "z"),
                ),
                sample_args=placement.get("sample_args", None),
            )

        return placement_initializer

    def _create_objects(self):
        """
        Creates and places objects in the kitchen environment.
        Helper function called by _load_model()
        """
        # add objects
        self.objects = {}
        if "object_cfgs" in self._ep_meta:
            self.object_cfgs = self._ep_meta["object_cfgs"]
            for obj_num, cfg in enumerate(self.object_cfgs):
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = self._create_obj(cfg)
                cfg["info"] = info
                self.objects[model.name] = model
        else:
            self.object_cfgs = self._get_obj_cfgs()
            addl_obj_cfgs = []
            for obj_num, cfg in enumerate(self.object_cfgs):
                cfg["type"] = "object"
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = self._create_obj(cfg)
                cfg["info"] = info
                self.objects[model.name] = model

                try_to_place_in = cfg["placement"].get("try_to_place_in", None)

                # place object in a container and add container as an object to the scene
                if try_to_place_in and (
                    "in_container" in cfg["info"]["groups_containing_sampled_obj"]
                ):
                    container_cfg = {
                        "name": cfg["name"] + "_container",
                        "obj_groups": cfg["placement"].get("try_to_place_in"),
                        "placement": deepcopy(cfg["placement"]),
                        "type": "object",
                    }

                    container_kwargs = cfg["placement"].get("container_kwargs", None)
                    if container_kwargs is not None:
                        for k, v in container_kwargs.items():
                            container_cfg[k] = v

                    # add in the new object to the model
                    addl_obj_cfgs.append(container_cfg)
                    model, info = self._create_obj(container_cfg)
                    container_cfg["info"] = info
                    self.objects[model.name] = model

                    # modify object config to lie inside of container
                    cfg["placement"] = dict(
                        size=(0.01, 0.01),
                        ensure_object_boundary_in_range=False,
                        sample_args=dict(
                            reference=container_cfg["name"],
                        ),
                    )

            # prepend the new object configs in
            self.object_cfgs = addl_obj_cfgs + self.object_cfgs

    def _create_obj(self, cfg):
        """
        Helper function for creating objects.
        Called by _create_objects()
        """
        if "info" in cfg:
            """
            if cfg has "info" key in it, that means it is storing meta data already
            that indicates which object we should be using.
            set the obj_groups to this path to do deterministic playback
            """
            mjcf_path = cfg["info"]["mjcf_path"]
            # replace with correct base path
            new_base_path = os.path.join(robocasa.models.assets_root, "objects")
            new_path = os.path.join(new_base_path, mjcf_path.split("/objects/")[-1])
            obj_groups = new_path
            exclude_obj_groups = None
        else:
            obj_groups = cfg.get("obj_groups", "all")
            exclude_obj_groups = cfg.get("exclude_obj_groups", None)
        object_kwargs, object_info = self.sample_object(
            obj_groups,
            exclude_groups=exclude_obj_groups,
            graspable=cfg.get("graspable", None),
            washable=cfg.get("washable", None),
            microwavable=cfg.get("microwavable", None),
            cookable=cfg.get("cookable", None),
            freezable=cfg.get("freezable", None),
            max_size=cfg.get("max_size", (None, None, None)),
            object_scale=cfg.get("object_scale", None),
        )
        info = object_info

        object = MJCFObject(name=cfg["name"], **object_kwargs)

        return object, info

    def _get_obj_cfgs(self):
        """
        Returns a list of object configurations to use in the environment.
        The object configurations are usually environment-specific and should
        be implemented in the subclass.

        Returns:
            list: list of object configurations
        """
        return []

    def sample_object(
        self,
        groups,
        exclude_groups=None,
        graspable=None,
        microwavable=None,
        washable=None,
        cookable=None,
        freezable=None,
        split=None,
        obj_registries=None,
        max_size=(None, None, None),
        object_scale=None,
    ):
        """
        Sample a kitchen object from the specified groups and within max_size bounds.

        Args:
            groups (list or str): groups to sample from or the exact xml path of the object to spawn

            exclude_groups (str or list): groups to exclude

            graspable (bool): whether the sampled object must be graspable

            washable (bool): whether the sampled object must be washable

            microwavable (bool): whether the sampled object must be microwavable

            cookable (bool): whether whether the sampled object must be cookable

            freezable (bool): whether whether the sampled object must be freezable

            split (str): split to sample from. Split "A" specifies all but the last 3 object instances
                        (or the first half - whichever is larger), "B" specifies the  rest, and None
                        specifies all.

            obj_registries (tuple): registries to sample from

            max_size (tuple): max size of the object. If the sampled object is not within bounds of max size,
                            function will resample

            object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value

        Returns:
            dict: kwargs to apply to the MJCF model for the sampled object

            dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to,
            the category of the object the sampling split the object came from, and the groups the object was sampled from
        """
        return sample_kitchen_object(
            groups,
            exclude_groups=exclude_groups,
            graspable=graspable,
            washable=washable,
            microwavable=microwavable,
            cookable=cookable,
            freezable=freezable,
            rng=self.rng,
            obj_registries=(obj_registries or self.obj_registries),
            split=(split or self.obj_instance_split),
            max_size=max_size,
            object_scale=object_scale,
        )

    def get_ep_meta(self):
        """
        Returns a dictionary containing episode meta data
        """
        def copy_dict_for_json(orig_dict):
            new_dict = {}
            for (k, v) in orig_dict.items():
                if isinstance(v, dict):
                    new_dict[k] = copy_dict_for_json(v)
                elif isinstance(v, Fixture):
                    new_dict[k] = v.name
                else:
                    new_dict[k] = v
            return new_dict

        ep_meta = {}
        ep_meta["layout_id"] = self.layout_id
        ep_meta["style_id"] = self.style_id
        ep_meta["object_cfgs"] = [copy_dict_for_json(cfg) for cfg in self.object_cfgs]
        ep_meta["fixtures"] = {
            k: {"cls": v.__class__.__name__} for (k, v) in self.fixtures.items()
        }
        ep_meta["gen_textures"] = self._curr_gen_fixtures or {}
        ep_meta["lang"] = ""
        ep_meta["fixture_refs"] = dict(
            {k: v.name for (k, v) in self.fixture_refs.items()}
        )

        return ep_meta

    def edit_model_xml(self, xml_str):
        """
        This function postprocesses the model.xml for retrospective model changes.

        Args:
            xml_str (str): Mujoco sim demonstration XML file as string

        Returns:
            str: Post-processed xml file as string
        """
        tree = ET.fromstring(xml_str)
        root = tree
        worldbody = root.find("worldbody")
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        robocasa_path_split = os.path.split(robocasa.__file__)[0].split("/")

        # Replace robocasa-specific asset paths
        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue

            old_path_split = old_path.split("/")
            # Maybe replace all paths to robosuite assets
            if (
                ("models/assets/fixtures" in old_path)
                or ("models/assets/textures" in old_path)
                or ("models/assets/objects/objaverse" in old_path)
            ):
                if "/robosuite/" in old_path:
                    check_lst = [
                        loc
                        for loc, val in enumerate(old_path_split)
                        if val == "robosuite"
                    ]
                elif "/robocasa/" in old_path:
                    check_lst = [
                        loc
                        for loc, val in enumerate(old_path_split)
                        if val == "robocasa"
                    ]
                else:
                    raise ValueError

                ind = max(check_lst)  # last occurrence index
                new_path_split = robocasa_path_split + old_path_split[ind + 1 :]

                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        result = ET.tostring(root).decode("utf8")

        # Apply texture overrides BEFORE generative textures
        if self.force_style_textures is not None:
            result = self._apply_style_overrides(result)

        # Replace with generative textures
        if (self.generative_textures is not None) and (
            self.generative_textures is not False
        ):
            # Sample textures
            assert self.generative_textures == "100p"
            if self._curr_gen_fixtures is None or self._curr_gen_fixtures == {}:
                self._curr_gen_fixtures = get_random_textures(self.rng)

            cab_tex = self._curr_gen_fixtures["cab_tex"]
            counter_tex = self._curr_gen_fixtures["counter_tex"]
            wall_tex = self._curr_gen_fixtures["wall_tex"]
            floor_tex = self._curr_gen_fixtures["floor_tex"]

            result = replace_cab_textures(
                self.rng, result, new_cab_texture_file=cab_tex
            )
            result = replace_counter_top_texture(
                self.rng, result, new_counter_top_texture_file=counter_tex
            )
            result = replace_wall_texture(
                self.rng, result, new_wall_texture_file=wall_tex
            )
            result = replace_floor_texture(
                self.rng, result, new_floor_texture_file=floor_tex
            )

        return result

    def _apply_style_overrides(self, xml_str):
        """
        Apply style-specific texture overrides using proper texture replacement functions.
        
        Args:
            xml_str (str): XML string to modify
            
        Returns:
            str: Modified XML string
        """
        if self.force_style_textures == "no_red_walls":
            # Use white brick texture instead of red
            white_brick_path = os.path.join(
                robocasa.models.assets_root, 
                "textures", "bricks", "white_bricks.png"
            )
            
            # Replace wall textures with white bricks
            xml_str = replace_wall_texture(
                self.rng, 
                xml_str, 
                new_wall_texture_file=white_brick_path
            )
            
            # Also replace counter textures to avoid red countertops
            white_marble_path = os.path.join(
                robocasa.models.assets_root, 
                "textures", "marble", "marble_2.png"
            )
            xml_str = replace_counter_top_texture(
                self.rng, 
                xml_str, 
                new_counter_top_texture_file=white_marble_path
            )
            
        return xml_str

    def _is_fxtr_valid(self, fxtr, size):
        """
        Checks if counter is valid for object placement by making sure it is large enough

        Args:
            fxtr (Fixture): fixture to check
            size (tuple): minimum size (x,y) that the counter region must be to be valid

        Returns:
            bool: True if fixture is valid, False otherwise
        """
        for region in fxtr.get_reset_regions(self).values():
            if region["size"][0] >= size[0] and region["size"][1] >= size[1]:
                return True
        return False

    def get_fixture(self, id, ref=None, size=(0.2, 0.2)):
        """
        Search fixture by id (name, object, or type)

        Args:
            id (str, Fixture, FixtureType): id of fixture to search for
            ref (str, Fixture, FixtureType): if specified, will search for fixture close to ref (within 0.10m)
            size (tuple): if sampling counter, minimum size (x,y) that the counter region must be

        Returns:
            Fixture: fixture object
        """
        # Case 1: id refers to fixture object directly
        if isinstance(id, Fixture):
            return id
        # Case 2: id refers to exact name of fixture
        elif id in self.fixtures.keys():
            return self.fixtures[id]

        if ref is None:
            # Find all fixtures with names containing given name
            if isinstance(id, FixtureType) or isinstance(id, int):
                matches = [
                    name
                    for (name, fxtr) in self.fixtures.items()
                    if fixture_is_type(fxtr, id)
                ]
            else:
                matches = [name for name in self.fixtures.keys() if id in name]
            if id == FixtureType.COUNTER or id == FixtureType.COUNTER_NON_CORNER:
                matches = [
                    name
                    for name in matches
                    if self._is_fxtr_valid(self.fixtures[name], size)
                ]
            assert len(matches) > 0
            # Sample random key
            key = self.rng.choice(matches)
            return self.fixtures[key]
        else:
            ref_fixture = self.get_fixture(ref)

            assert isinstance(id, FixtureType)
            cand_fixtures = []
            for fxtr in self.fixtures.values():
                if not fixture_is_type(fxtr, id):
                    continue
                if fxtr is ref_fixture:
                    continue
                if id == FixtureType.COUNTER:
                    fxtr_is_valid = self._is_fxtr_valid(fxtr, size)
                    if not fxtr_is_valid:
                        continue
                cand_fixtures.append(fxtr)

            # First, try to find fixture "containing" the reference fixture
            for fxtr in cand_fixtures:
                if OU.point_in_fixture(ref_fixture.pos, fxtr, only_2d=True):
                    return fxtr
            # If no fixture contains reference fixture, sample all close fixtures
            dists = [
                OU.fixture_pairwise_dist(ref_fixture, fxtr) for fxtr in cand_fixtures
            ]
            min_dist = np.min(dists)
            close_fixtures = [
                fxtr for (fxtr, d) in zip(cand_fixtures, dists) if d - min_dist < 0.10
            ]
            return self.rng.choice(close_fixtures)

    def register_fixture_ref(self, ref_name, fn_kwargs):
        """
        Registers a fixture reference for later use. Initializes the fixture
        if it has not been initialized yet.

        Args:
            ref_name (str): name of the reference
            fn_kwargs (dict): keyword arguments to pass to get_fixture

        Returns:
            Fixture: fixture object
        """
        if ref_name not in self.fixture_refs:
            self.fixture_refs[ref_name] = self.get_fixture(**fn_kwargs)
        return self.fixture_refs[ref_name]

    def get_arena_xml(self):
        """
        Get the arena XML string.
        
        Returns:
            str: XML string of the kitchen arena
        """
        xml_str = self.mujoco_arena.get_xml()
        return self.edit_model_xml(xml_str)
    
    def get_complete_model_xml(self):
        """
        Get the complete kitchen model XML including all fixtures.
        Uses ManipulationTask to properly merge components and avoid layout issues.
        
        Returns:
            str: XML string of the complete kitchen model
        """
        
        
        # Create the task model with no robots, just fixtures and objects
        task_model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[],  # No robots in simple version
            mujoco_objects=list(self.fixtures.values()) + list(self.objects.values()),
        )
        
        # Get the complete XML and apply post-processing
        xml_str = self.edit_model_xml(task_model.get_xml())
        return xml_str