"""
Standalone MjSim class extracted from robosuite.
This provides a simplified simulation interface for MuJoCo models.
"""

import mujoco
import numpy as np


class MjSimState:
    """
    A mujoco simulation state.
    """

    def __init__(self, time, qpos, qvel):
        self.time = time
        self.qpos = qpos
        self.qvel = qvel

    @classmethod
    def from_flattened(cls, array, sim):
        """
        Takes flat mjstate array and MjSim instance and
        returns MjSimState.
        """
        idx_time = 0
        idx_qpos = idx_time + 1
        idx_qvel = idx_qpos + sim.model.nq

        time = array[idx_time]
        qpos = array[idx_qpos : idx_qpos + sim.model.nq]
        qvel = array[idx_qvel : idx_qvel + sim.model.nv]

        return cls(time=time, qpos=qpos, qvel=qvel)

    def flatten(self):
        return np.concatenate([[self.time], self.qpos, self.qvel], axis=0)


class MjModel:
    """Wrapper class for a MuJoCo 'mjModel' instance."""

    def __init__(self, model_ptr):
        """Creates a new MjModel instance from a mujoco.MjModel."""
        self._model = model_ptr

        # Create name to id mappings for convenience
        self._body_name2id = {name: i for i, name in enumerate(self.body_names)}
        self._body_id2name = {i: name for name, i in self._body_name2id.items()}
        self._joint_name2id = {name: i for i, name in enumerate(self.joint_names)}
        self._joint_id2name = {i: name for name, i in self._joint_name2id.items()}
        self._geom_name2id = {name: i for i, name in enumerate(self.geom_names)}
        self._geom_id2name = {i: name for name, i in self._geom_name2id.items()}
        self._actuator_name2id = {name: i for i, name in enumerate(self.actuator_names)}
        self._actuator_id2name = {i: name for name, i in self._actuator_name2id.items()}
        self._camera_name2id = {name: i for i, name in enumerate(self.camera_names)}
        self._camera_id2name = {i: name for name, i in self._camera_name2id.items()}
        self._site_name2id = {name: i for i, name in enumerate(self.site_names)}
        self._site_id2name = {i: name for name, i in self._site_name2id.items()}

    def __del__(self):
        # free mujoco model
        del self._model

    # Expose common properties
    @property
    def nq(self):
        return self._model.nq

    @property
    def nv(self):
        return self._model.nv

    @property
    def na(self):
        return self._model.na

    @property
    def nu(self):
        return self._model.nu

    @property
    def nbody(self):
        return self._model.nbody

    @property
    def njnt(self):
        return self._model.njnt

    @property
    def ngeom(self):
        return self._model.ngeom

    @property
    def nsite(self):
        return self._model.nsite

    @property
    def ncam(self):
        return self._model.ncam

    @property 
    def body_names(self):
        return [self._model.body(i).name for i in range(self.nbody)]

    @property
    def joint_names(self):
        return [self._model.joint(i).name for i in range(self.njnt)]

    @property
    def geom_names(self):
        return [self._model.geom(i).name for i in range(self.ngeom)]

    @property
    def actuator_names(self):
        return [self._model.actuator(i).name for i in range(self.nu)]

    @property
    def camera_names(self):
        return [self._model.camera(i).name for i in range(self.ncam)]

    @property
    def site_names(self):
        return [self._model.site(i).name for i in range(self.nsite)]

    def body_name2id(self, name):
        return self._body_name2id.get(name, -1)

    def body_id2name(self, id):
        return self._body_id2name.get(id, "")

    def joint_name2id(self, name):
        return self._joint_name2id.get(name, -1)

    def joint_id2name(self, id):
        return self._joint_id2name.get(id, "")

    def geom_name2id(self, name):
        return self._geom_name2id.get(name, -1)

    def geom_id2name(self, id):
        return self._geom_id2name.get(id, "")

    def actuator_name2id(self, name):
        return self._actuator_name2id.get(name, -1)

    def actuator_id2name(self, id):
        return self._actuator_id2name.get(id, "")

    def camera_name2id(self, name):
        return self._camera_name2id.get(name, -1)

    def camera_id2name(self, id):
        return self._camera_id2name.get(id, "")

    def site_name2id(self, name):
        return self._site_name2id.get(name, -1)

    def site_id2name(self, id):
        return self._site_id2name.get(id, "")


class MjData:
    """Wrapper class for a MuJoCo 'mjData' instance."""

    def __init__(self, model):
        """Construct a new MjData instance.
        Args:
          model: An MjModel instance.
        """
        self._model = model
        self._data = mujoco.MjData(model._model)

    @property
    def model(self):
        """The parent MjModel for this MjData instance."""
        return self._model

    def __del__(self):
        # free mujoco data
        del self._data

    @property
    def time(self):
        return self._data.time

    @time.setter
    def time(self, value):
        self._data.time = value

    @property
    def qpos(self):
        return self._data.qpos

    @property
    def qvel(self):
        return self._data.qvel

    @property
    def ctrl(self):
        return self._data.ctrl

    # Support body_xpos, body_xquat, body_xmat for compatibility
    @property
    def body_xpos(self):
        return self._data.xpos

    @property
    def body_xquat(self):
        return self._data.xquat

    @property
    def body_xmat(self):
        return self._data.xmat


class MjSim:
    """
    Simplified MjSim object based on robosuite implementation.
    """

    def __init__(self, model):
        """
        Args:
            model: should be an MjModel instance created via a factory function
                such as mujoco.MjModel.from_xml_string(xml)
        """
        self.model = MjModel(model)
        self.data = MjData(self.model)

    @classmethod
    def from_xml_string(cls, xml):
        model = mujoco.MjModel.from_xml_string(xml)
        return cls(model)

    @classmethod
    def from_xml_file(cls, xml_file):
        f = open(xml_file, "r")
        xml = f.read()
        f.close()
        return cls.from_xml_string(xml)

    def reset(self):
        """Reset simulation."""
        mujoco.mj_resetData(self.model._model, self.data._data)

    def forward(self):
        """Forward call to synchronize derived quantities."""
        mujoco.mj_forward(self.model._model, self.data._data)

    def step(self):
        """Step simulation."""
        mujoco.mj_step(self.model._model, self.data._data)

    def step1(self):
        """Step1 (before actions are set)."""
        mujoco.mj_step1(self.model._model, self.data._data)

    def step2(self):
        """Step2 (after actions are set)."""
        mujoco.mj_step2(self.model._model, self.data._data)

    def get_state(self):
        """Return MjSimState instance for current state."""
        return MjSimState(
            time=self.data.time,
            qpos=np.copy(self.data.qpos),
            qvel=np.copy(self.data.qvel),
        )

    def set_state(self, value):
        """
        Set internal state from MjSimState instance. Should
        call @forward afterwards to synchronize derived quantities.
        """
        self.data.time = value.time
        self.data.qpos[:] = np.copy(value.qpos)
        self.data.qvel[:] = np.copy(value.qvel)

    def set_state_from_flattened(self, value):
        """
        Set internal mujoco state using flat mjstate array. Should
        call @forward afterwards to synchronize derived quantities.
        """
        state = MjSimState.from_flattened(value, self)

        # do this instead of @set_state to avoid extra copy of qpos and qvel
        self.data.time = state.time
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel 