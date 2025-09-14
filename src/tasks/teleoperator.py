"""
SO101 Kitchen Teleoperation Controller

Simple teleoperation controller for SO101 robotic arm in RoboCasa kitchen environment.
Maps physical leader arm movements to follower arm to pick banana and place in bowl.

Usage:
    conda activate lerobot
    python main.py --task BananaToBowlPnP --collect-dataset --record-video
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
import time
import xml.etree.ElementTree as ET
from typing import Dict, Any

from robot.teleop.so101_leader import SO101Leader, create_leader_config
from utils.dataset_collector import DatasetCollector


class SO101TeleoperationController:
    """Main controller class for SO101 kitchen teleoperation."""
    
    JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    # Joint control ranges from so101_simplified.xml
    CTRL_RANGES = {
        "shoulder_pan": (-1.91986, 1.91986),
        "shoulder_lift": (-1.74533, 1.74533),
        "elbow_flex": (-1.69, 1.69),
        "wrist_flex": (-1.65806, 1.65806),
        "wrist_roll": (-2.74385, 2.84121),
        "gripper": (-0.17453, 1.74533)
    }

    def __init__(self, port, calibration_file, robot_xml_path, layout_id=0, style_id=0, task_name="BananaToBowlPnP", fps=30):
        self.port = port
        self.calibration_file = calibration_file
        self.robot_xml_path = robot_xml_path
        self.layout_id = layout_id
        self.style_id = style_id
        self.task_name = task_name
        self._task_class = None
        
        self.leader = None
        self.kitchen_env = None
        self.model = None
        self.data = None
        self.scene_xml_path = None
        
        # Control parameters
        self.control_freq = fps  # Hz
        self.control_period = 1.0 / self.control_freq
        
        # Joint and actuator mappings
        self.joint_ids = {}
        self.joint_qpos_addrs = {}
        self.actuator_ids = {}
        
        # Dataset collection
        self.dataset_collector = None
        self._offscreen_scene_option = None
        
        # Recording parameters
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fps = self.control_freq  # Match teleoperation frequency
        self.recording_timeout = 10  # Default timeout in seconds
    
    def initialize_hardware(self):
        """Initialize the SO101 leader arm hardware interface."""
        print("Initializing hardware interface...")
        
        # Preferred local unified teleop
        leader_config = create_leader_config(
            port=self.port,
            calibration_file=os.path.basename(self.calibration_file),
            calibration_dir=os.path.dirname(self.calibration_file) or None,
        )
        self.leader = SO101Leader(leader_config)
        self.leader.connect(calibrate=False) # calibrate=True to calibrate the robot

    def initialize_kitchen_environment(self):
        """Initialize the RoboCasa kitchen environment for the selected task.
        Must be overridden by task-specific teleop subclasses.
        """
        raise NotImplementedError("initialize_kitchen_environment must be overridden by the task teleop class")
    
    def integrate_robot_into_kitchen(self):
        """Integrate SO101 robot into the RoboCasa kitchen environment XML."""
        print("Integrating SO101 robot into kitchen environment...")
        
        if not os.path.exists(self.robot_xml_path):
            raise FileNotFoundError(f"Robot XML file not found: {self.robot_xml_path}")
        
        # Get the complete kitchen XML
        kitchen_xml_str = self.kitchen_env.get_complete_model_xml()
        
        # Save kitchen XML to temporary file
        temp_kitchen_path = "temp_kitchen_scene.xml"
        with open(temp_kitchen_path, "w") as f:
            f.write(kitchen_xml_str)
        
        try:
            # Load kitchen and robot XMLs
            from mujoco_utils.mujoco_xml import MujocoXML
            kitchen_xml = MujocoXML(temp_kitchen_path)
            robot_xml = MujocoXML(self.robot_xml_path)
            
            # Position robot on counter surface and capture base body
            robot_base = self._position_robot_base(kitchen_xml, robot_xml)
            
            # Add overhead camera for scene overview
            self._add_overhead_camera(kitchen_xml, robot_base)
            
            
            if self.task_name == "BananaToBowlPnP":
                self._add_side_camera(kitchen_xml, robot_base)
                
            
            # Add base stand
            if self.task_name == "Drawer" or self.task_name == "MicrowaveDoor":
                self._add_base_stand(kitchen_xml, robot_base)
            # self._add_base_stand(kitchen_xml, robot_base)
            
            # Merge robot into kitchen scene
            kitchen_xml.merge(robot_xml)
        

            # Apply MuJoCo scene options (enable multiccd etc.)
            self._apply_scene_options(kitchen_xml)
            
            # Save the merged scene under configs/kitchen_configs
            configs_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "configs", "kitchen_configs")
            )
            os.makedirs(configs_dir, exist_ok=True)
            self.scene_xml_path = os.path.join(configs_dir, "kitchen_robot_scene.xml")
            kitchen_xml.save_model(self.scene_xml_path)
            
            print(f"Saved merged scene file: {self.scene_xml_path}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_kitchen_path):
                os.remove(temp_kitchen_path)

    def _apply_scene_options(self, kitchen_xml):
        """Apply MuJoCo <option> settings to the merged scene for smoother grasping."""
        root = kitchen_xml.root

        # Ensure <option><flag/> exists and set smoother grasping parameters
        option = root.find(".//option")
        if option is None:
            option = ET.SubElement(root, "option")
            
        # Smoother grasping parameters
        option.set("noslip_iterations", "3")  # Increased for better stability
        
        flag = option.find("flag")
        if flag is None:
            flag = ET.SubElement(option, "flag")
        flag.set("multiccd", "enable")
        flag.set("nativeccd", "enable")
        flag.set("energy", "enable")
        flag.set("gravity", "enable")
    
    def initialize_mujoco_simulation(self):
        """Initialize MuJoCo simulation with the merged scene."""
        print("Setting up MuJoCo simulation...")
        
        # Try new MuJoCo API (3.x)
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Apply object placements from kitchen environment
        self._apply_object_placements()
        
        # Build joint and actuator index maps
        self._build_joint_mappings()
        
        # Set initial robot pose
        self._set_initial_robot_pose()
        
        # Replace banana with orange
        # self._replace_object_with_object("banana", size=0.03, mass=0.2, friction="1 0.5 0.1", rgba="1 0.5 0 1")
        
        # Move objects to designated positions after scene creation
        self._move_objects_to_designated_positions()
        
        # Prepare offscreen rendering scene option
        self._prepare_offscreen_rendering()
        
        print(f"Simulation ready: {self.model.njnt} joints, {self.model.nu} actuators")
    
    def _move_objects_to_designated_positions(self):
        """Move objects to their designated positions relative to robot base."""
        if self.task_name == "BananaToBowlPnP" or self.task_name == "OrangeToPlatePnP" or self.task_name == "PanToSinkPnP":
            print("Moving objects to designated positions...")
            
            # Get robot base position
            robot_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            if robot_base_id == -1:
                print("Warning: Robot base body not found")
                return
            
            robot_base_pos = self.data.xpos[robot_base_id].copy()
            print(f"Robot base position: {robot_base_pos}")
            
            # First, find the actual joint names used for the objects
            object_joint_mapping = {}
            if hasattr(self.kitchen_env, 'object_placements') and self.kitchen_env.object_placements:
                for obj_name, (pos, quat, obj) in self.kitchen_env.object_placements.items():
                    if obj_name in ["banana", "bowl", "apple", "can_4", "plate", "mango", "peach", "orange", "pan", "sponge"]:
                        joint_name = obj.joints[0]
                        object_joint_mapping[obj_name] = joint_name
                        print(f"Found {obj_name} with joint name: {joint_name}")
            
            # Define target positions and orientations relative to robot base
            target_positions = {
                "banana": robot_base_pos + np.array([0.2, 0.3, 0.0]),
                "apple": robot_base_pos + np.array([0.2, 0.1, 0.0]),
                "can_4": robot_base_pos + np.array([-0.2, 0.1, 0.1]),
                "plate": robot_base_pos + np.array([-0.2, 0.2, 0.0]),
                "mango": robot_base_pos + np.array([-0.2, 0.3, 0.05]), 
                "peach": robot_base_pos + np.array([-0.2, 0.2, 0.0]),
                "orange": robot_base_pos + np.array([0.2, 0.2, 0.0]),
                "pan": robot_base_pos + np.array([0.2, 0.4, 0.0]),
                "sponge": robot_base_pos + np.array([-0.1, 0.1, 0.0]),
            }
            
            # Define target orientations (quaternions: w, x, y, z)
            # Banana rotated 90 degrees around yaw axis (Z-axis)
            target_orientations = {
                "banana": np.array([0.707, 0.0, 0.0, 0.707]),  # 90° rotation around Z-axis (yaw)
                # "banana": np.array([1.0, 0.0, 0.0, 0.0]),  # No rotation
                "bowl": np.array([1.0, 0.0, 0.0, 0.0]),         # No rotation
                "pan": np.array([0.9659, 0, 0, -0.2588]),   # -45 degrees around Z-axis (yaw)
                "sponge": np.array([0.9659, 0, 0, -0.2588]), # -45 degrees around Z-axis (yaw)
            }
            
            
            # Define target sizes for objects (x, y, z dimensions)
            target_sizes = {
                "banana": "0.03 0.03 0.08",      # Make banana smaller
                "bowl": "0.08 0.08 0.04",        # Make bowl smaller
                "orange": "0.04 0.04 0.04",      # Make orange smaller
                "apple": "0.04 0.04 0.04",       # Make apple smaller
                "plate": "0.12 0.12 0.01",       # Make plate smaller and flatter
                "mango": "0.05 0.05 0.08",       # Make mango smaller
                "peach": "0.04 0.04 0.05",       # Make peach smaller
                "can_4": "0.03 0.03 0.08"        # Make can smaller
            }
            
            # Move each object to its target position and orientation using the correct joint name
            for obj_name, target_pos in target_positions.items():
                if obj_name in object_joint_mapping:
                    joint_name = object_joint_mapping[obj_name]
                    try:
                        # Find the object's joint using the correct joint name
                        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                        if joint_id != -1:
                            adr = self.model.jnt_qposadr[joint_id]
                            
                            # Set new position
                            self.data.qpos[adr:adr+3] = target_pos
                            
                            # Set new orientation if specified
                            if obj_name in target_orientations:
                                target_quat = target_orientations[obj_name]
                                self.data.qpos[adr+3:adr+7] = target_quat
                                orientation_info = f" with orientation {target_quat}"
                            else:
                                # Preserve current orientation if no target specified
                                current_quat = self.data.qpos[adr+3:adr+7].copy()
                                self.data.qpos[adr+3:adr+7] = current_quat
                                orientation_info = " (preserved orientation)"
                                
                            # Set new size if specified
                            if obj_name in target_sizes:
                                from mujoco_utils.mujoco_xml import MujocoXML
                                kitchen_xml = MujocoXML(self.scene_xml_path)
                                object_body = kitchen_xml.root.find(f".//body[@name='{obj_name}']")
                                if object_body is not None:
                                    geom = object_body.find(".//geom")
                                    if geom is not None:
                                        geom.set("size", target_sizes[obj_name])
                                        kitchen_xml.save_model(self.scene_xml_path)
                            
                            print(f"Moved {obj_name} (joint: {joint_name}) to position {target_pos}{orientation_info}")
                        else:
                            print(f"Warning: Joint {joint_name} not found for {obj_name}")
                    except Exception as e:
                        print(f"Error moving {obj_name}: {e}")
                else:
                    print(f"Warning: No joint mapping found for {obj_name}")
            
            # Update forward kinematics
            mujoco.mj_forward(self.model, self.data)
            print("Object positioning complete")
    
    def _prepare_offscreen_rendering(self):
        """Prepare scene option for offscreen rendering (hide collision meshes)."""
        self._offscreen_scene_option = mujoco.MjvOption()
        self._offscreen_scene_option.geomgroup[0] = 0  # Hide collision meshes
        for i in range(1, 6):  # Show visual groups 1-5
            self._offscreen_scene_option.geomgroup[i] = 1
    
    def _apply_object_placements(self):
        """Apply object placements from kitchen environment and save the final scene."""
        if hasattr(self.kitchen_env, 'object_placements') and self.kitchen_env.object_placements:
            print(f"Applying {len(self.kitchen_env.object_placements)} object placements...")
            for obj_name, (pos, quat, obj) in self.kitchen_env.object_placements.items():
                try:
                    joint_name = obj.joints[0]
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if jid != -1:
                        adr = self.model.jnt_qposadr[jid]
                        self.data.qpos[adr:adr+7] = np.concatenate([np.array(pos), np.array(quat)])
                        print(f"Placed {obj_name} at position {pos}")
                    else:
                        print(f"Joint {joint_name} not found for {obj_name}")
                except Exception as e:
                    print(f"Could not place {obj_name}: {e}")
            
            # Forward kinematics to update positions
            mujoco.mj_forward(self.model, self.data)
            print("Object placements applied")

            # Save the final scene with object placements
            try:
                mujoco.mj_saveLastXML(self.scene_xml_path, self.model, None)
                print(f"Saved final scene with object placements to {self.scene_xml_path}")
            except Exception as e:
                print(f"Failed to save scene XML: {e}")

        else:
            print("No object placements found in kitchen environment")
    
    
    def _build_joint_mappings(self):
        """Build joint and actuator index maps by name."""
        for joint_name in self.JOINT_ORDER:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid != -1:
                self.joint_ids[joint_name] = jid
                self.joint_qpos_addrs[joint_name] = self.model.jnt_qposadr[jid]
        
        for joint_name in self.JOINT_ORDER:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if aid != -1:
                self.actuator_ids[joint_name] = aid
    
    def _set_initial_robot_pose(self):
        """Set initial robot pose based on leader arm position."""
        print("Setting initial robot pose...")
        initial_leader_action = self.leader.get_action()
        print(f"Initial leader action: {initial_leader_action}")
        initial_follower_pos = self._map_leader_to_follower(initial_leader_action)
        print(f"Mapped follower positions: {initial_follower_pos}")
        
        for i, joint_name in enumerate(self.JOINT_ORDER):
            if joint_name in self.joint_qpos_addrs:
                adr = self.joint_qpos_addrs[joint_name]
                old_pos = self.data.qpos[adr]
                self.data.qpos[adr] = initial_follower_pos[i]
                print(f"Joint {joint_name}: {old_pos} -> {initial_follower_pos[i]}")
        
        mujoco.mj_forward(self.model, self.data)
        print("Initial robot pose set")
    
    def _normalize_leader_action(self, leader_action):
        """Normalize leader action into expected ranges."""
        positions = []
        for joint in self.JOINT_ORDER:
            key = f"{joint}.pos"
            positions.append(leader_action.get(key, 0.0))
        positions = np.array(positions, dtype=float)
        
        # Detect raw counts (e.g., Feetech 0-4095) vs normalized
        if np.max(np.abs(positions)) > 1000:
            # Convert counts to normalized scale
            counts = np.clip(positions, 0.0, 4095.0)
            normalized = np.zeros_like(counts)
            mid = 2047.5
            
            for i in range(len(counts)):
                if self.JOINT_ORDER[i] == "gripper":
                    normalized[i] = (counts[i] / 4095.0) * 100.0
                else:
                    normalized[i] = ((counts[i] - mid) / mid) * 100.0
            return normalized
        else:
            # Already normalized
            return positions
    
    def _map_leader_to_follower(self, leader_action):
        """Map SO101 leader joint positions to MuJoCo actuator control range."""
        positions = self._normalize_leader_action(leader_action)
        mapped_positions = np.zeros_like(positions)
        
        for i, joint_name in enumerate(self.JOINT_ORDER):
            ctrl_min, ctrl_max = self.CTRL_RANGES[joint_name]
            
            if joint_name == "gripper":
                # Gripper joint: [0, 100] to [ctrl_min, ctrl_max]
                normalized = np.clip(positions[i] / 100.0, 0.0, 1.0)
                mapped_positions[i] = normalized * (ctrl_max - ctrl_min) + ctrl_min
            else:
                # Arm joints: lerobot normalized values [-100, 100] to ctrl range
                clamped = np.clip(positions[i], -100.0, 100.0)
                normalized = (clamped + 100.0) / 200.0
                mapped_positions[i] = normalized * (ctrl_max - ctrl_min) + ctrl_min
        
        return mapped_positions
    
    def start_dataset_collection(self, output_dir: str = "collected_datasets"):
        """Start collecting dataset during teleoperation."""
        # Get task description
        try:
            ep_meta = self.kitchen_env.get_ep_meta()
            task_description = ep_meta.get('lang', self.task_name)
        except Exception:
            task_description = self.task_name
        
        # Initialize dataset collector
        self.dataset_collector = DatasetCollector(output_dir, fps=self.control_freq)
        self.dataset_collector.start_collection(task_description)
        
        # Setup video recording
        camera_names = ["wrist_camera", "overhead_camera"]
        self.dataset_collector.setup_video_recording(
            self.model, camera_names, 
            width=self.camera_width, 
            height=self.camera_height, 
            fps=self.camera_fps
        )
        
        print(f"Started dataset collection in {output_dir}")
        print(f"Task: {task_description}")
    
    def _collect_frame_data(self, leader_action: Dict[str, Any], follower_control: np.ndarray):
        """Collect data for current frame."""
        if self.dataset_collector is None:
            return
        
        # Get follower joint positions
        follower_joint_positions = []
        for joint_name in self.JOINT_ORDER:
            if joint_name in self.joint_qpos_addrs:
                adr = self.joint_qpos_addrs[joint_name]
                follower_joint_positions.append(self.data.qpos[adr])
        
        # Collect frame data
        self.dataset_collector.collect_frame(
            leader_action, follower_joint_positions, 
            self.data, self._offscreen_scene_option
        )
    
    def stop_dataset_collection(self):
        """Stop dataset collection and save final episode."""
        if self.dataset_collector is None:
            return
        
        self.dataset_collector.stop_collection()
        self.dataset_collector = None
        print("Dataset collection stopped")
    
    def reset_environment(self):
        """Reset the environment to initial state."""
        print("Resetting environment...")
        
        # Reset simulation to initial state
        mujoco.mj_resetData(self.model, self.data)
        
        # Reapply object placements
        self._apply_object_placements()
        
        # Reset robot to initial pose
        self._set_initial_robot_pose()
        
        print("Environment reset complete")
    
    def run_teleoperation(self, dataset_output_dir="collected_datasets", recording_timeout=10):
        """Run the main teleoperation loop with keyboard-controlled dataset collection."""
        self.recording_timeout = recording_timeout
        print(f"\nStarting teleoperation at {self.control_freq} Hz")
        print("Move your leader arm to control the robot")
        print(f"\nRecording timeout: {self.recording_timeout} seconds")
        print("\nKeyboard Controls:")
        print("  S - Start/Stop recording episode")
        print("  R - Reset environment")
        print("  T - Save episode (when prompted)")
        print("  D - Discard episode (when prompted)")
        print("  Q - Quit teleoperation")
        print("  Ctrl+C - Emergency quit")
        
        # Try to print task language, fall back to task name
        try:
            ep_meta = self.kitchen_env.get_ep_meta()
            print(f"Task: {ep_meta.get('lang', self.task_name)}")
        except Exception:
            print(f"Task: {self.task_name}")
        
        # Initialize dataset collector but don't start collecting yet
        self.dataset_collector = DatasetCollector(dataset_output_dir)
        
        # Get task description for dataset
        try:
            ep_meta = self.kitchen_env.get_ep_meta()
            task_description = ep_meta.get('lang', self.task_name)
        except Exception:
            task_description = self.task_name
        
        # Setup video recording capabilities
        camera_names = ["wrist_camera", "overhead_camera"]
        self.dataset_collector.setup_video_recording(
            self.model, camera_names, 
            width=self.camera_width, 
            height=self.camera_height, 
            fps=self.camera_fps
        )
        
        # Set recording timeout
        self.dataset_collector.recording_timeout = self.recording_timeout
        
        # Start keyboard handler
        self.dataset_collector.start_keyboard_handler()
        
        # State tracking
        is_recording = False
        recording_start_time = None
        
        print(f"\nReady! Press 'S' to start recording your first episode...")
        
        # Main control loop with MuJoCo viewer
        with mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False) as viewer:
            # Configure viewer
            print("Configuring MuJoCo viewer...")
            viewer.opt.geomgroup[0] = 0  # Disable collision meshes
            for i in range(1, 6):  # Enable visual groups 1-5
                viewer.opt.geomgroup[i] = 1
            viewer.opt.frame = 0  # Disable frames (coordinate axes)
            viewer.opt.label = 0  # Disable labels (object names)
            
            # Print some debug info about the scene
            print(f"Scene has {self.model.ngeom} geometries")
            print(f"Scene has {self.model.nbody} bodies")
            print(f"Scene has {self.model.njnt} joints")
            print(f"Scene has {self.model.ncam} cameras")
            
            # List camera names
            for i in range(self.model.ncam):
                cam_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                print(f"Camera {i}: {cam_name}")
            
            # Set camera to overhead if available
            try:
                overhead_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_camera")
                if overhead_cam_id != -1:
                    viewer.cam.fixedcamid = overhead_cam_id
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    print("Set viewer to overhead camera")
            except:
                print("Could not set overhead camera, using free camera")
            
            try:
                while viewer.is_running():
                    start_time = time.time()
                    
                    # Handle keyboard commands
                    if self.dataset_collector.should_quit:
                        print("\nQuitting teleoperation...")
                        # Save current episode if recording
                        if is_recording:
                            print("Saving current episode before quit...")
                            self.dataset_collector.save_episode()
                            is_recording = False
                        break
                    
                    if self.dataset_collector.should_reset_episode:
                        print("Processing reset request...")
                        self.dataset_collector.should_reset_episode = False
                        self.reset_environment()
                        if is_recording:
                            print("Note: Recording is still active after reset")
                    
                    if self.dataset_collector.should_toggle_recording:
                        print("Processing recording toggle request...")
                        self.dataset_collector.should_toggle_recording = False
                        if not is_recording:
                            # Start recording
                            self.dataset_collector.start_collection(task_description)
                            is_recording = True
                            recording_start_time = time.time()
                            print(f"🔴 RECORDING STARTED - Episode {self.dataset_collector.episode_index}")
                            print(f"⏰ Recording will auto-stop in {self.recording_timeout} seconds")
                        else:
                            # Stop recording and ask for save decision
                            is_recording = False
                            recording_start_time = None
                            current_episode = self.dataset_collector.episode_index
                            frame_count = len(self.dataset_collector.episode_data)
                            print(f"⏹️  RECORDING STOPPED - Episode {current_episode} ({frame_count} frames)")
                            print("🤔 Do you want to save this episode?")
                            print("   Press 'T' to SAVE or 'D' to DISCARD")
                            self.dataset_collector.waiting_for_save_decision = True
                    
                    # Handle save/discard decision
                    if self.dataset_collector.should_save_episode:
                        self.dataset_collector.should_save_episode = False
                        self.dataset_collector.save_episode()
                        print(f"✅ Episode {self.dataset_collector.episode_index - 1} saved successfully!")
                        print("Press 'S' to start recording next episode or 'R' to reset environment")
                    
                    if self.dataset_collector.should_discard_episode:
                        self.dataset_collector.should_discard_episode = False
                        self.dataset_collector.discard_current_episode_data()
                        print("🗑️ Episode discarded successfully!")
                        print("Press 'S' to start recording again or 'R' to reset environment")
                    
                    # Check for recording timeout
                    if is_recording and recording_start_time is not None:
                        elapsed_time = time.time() - recording_start_time
                        remaining_time = self.recording_timeout - elapsed_time
                        
                        # Show countdown every second
                        if int(elapsed_time) != int(elapsed_time - self.control_period):
                            if remaining_time > 0:
                                print(f"⏰ Recording time remaining: {remaining_time:.1f}s", end='\r')
                        
                        # Auto-stop when timeout reached
                        if elapsed_time >= self.recording_timeout:
                            is_recording = False
                            recording_start_time = None
                            current_episode = self.dataset_collector.episode_index
                            frame_count = len(self.dataset_collector.episode_data)
                            print(f"\n⏰ RECORDING AUTO-STOPPED - Episode {current_episode} ({frame_count} frames)")
                            print(f"🕐 Timeout reached ({self.recording_timeout} seconds)")
                            print("🤔 Do you want to save this episode?")
                            print("   Press 'T' to SAVE or 'D' to DISCARD")
                            self.dataset_collector.waiting_for_save_decision = True
                    
                    # Get current leader action from hardware
                    leader_action = self.leader.get_action()
                    
                    # Map to follower control
                    follower_control = self._map_leader_to_follower(leader_action)
                    
                    # Apply control to robot actuators
                    for i, joint_name in enumerate(self.JOINT_ORDER):
                        aid = self.actuator_ids.get(joint_name, None)
                        if aid is not None and aid < self.data.ctrl.size:
                            self.data.ctrl[aid] = follower_control[i]
                    
                    # Step simulation
                    for _ in range(10):  # Multiple physics steps per control step
                        mujoco.mj_step(self.model, self.data)
                    
                    # Collect dataset frame if recording
                    if is_recording:
                        self._collect_frame_data(leader_action, follower_control)
                    
                    # Update viewer
                    viewer.sync()
                    
                    # Maintain control frequency
                    elapsed = time.time() - start_time
                    time.sleep(max(0, self.control_period - elapsed))
                    
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received - stopping teleoperation...")
                # Save current episode if recording
                if is_recording:
                    print("Saving current episode before exit...")
                    self.dataset_collector.save_episode()
                    is_recording = False
            finally:
                print("Performing final cleanup...")
                # Generate meta files and cleanup dataset collector
                if self.dataset_collector:
                    self.dataset_collector.cleanup()
                    print("Dataset collection completed and meta files generated")
                
                # Force cleanup of viewer and simulation
                print("Shutting down simulation...")
    
    def cleanup(self):
        """Clean up resources and temporary artifacts."""
        print("Starting teleoperator cleanup...")
        
        # Stop dataset collection if active and generate meta files
        if self.dataset_collector is not None:
            print("Cleaning up dataset collection...")
            try:
                self.dataset_collector.cleanup()
            except Exception as e:
                print(f"Warning: Error during dataset cleanup: {e}")
            finally:
                self.dataset_collector = None
            
        # Disconnect hardware
        if self.leader:
            print("Disconnecting hardware...")
            try:
                self.leader.disconnect()
            except Exception as e:
                print(f"Warning: Error disconnecting hardware: {e}")
            finally:
                self.leader = None

        # Remove temp kitchen file if we created one
        if os.path.exists("temp_kitchen_scene.xml"):
            try:
                os.remove("temp_kitchen_scene.xml")
                print("Removed temporary kitchen scene file")
            except Exception as e:
                print(f"Warning: Could not remove temp file: {e}")
        
        # Clear MuJoCo model and data references
        self.model = None
        self.data = None
        
        print("Teleoperator cleanup completed")
    
    def _replace_object_with_object(self, object_name, size=0.015, mass=0.5, friction="10 3 1.5", rgba="0 1 0 1"):
        # Find the object body
        from mujoco_utils.mujoco_xml import MujocoXML
        kitchen_xml = MujocoXML(self.scene_xml_path)
        
        previous_object_body = kitchen_xml.root.find(f".//body[@name='{object_name}']")
        if previous_object_body is None:
            print(f"Warning: {object_name} body not found")
            return

        # Remove existing object geoms (keep the joint + site)
        for geom in list(previous_object_body.findall("geom")):
            previous_object_body.remove(geom)
            
        # Remove existing inertial
        for inertial in list(previous_object_body.findall("inertial")):
            previous_object_body.remove(inertial)
            
        # Add new inertial
        inertial = ET.SubElement(previous_object_body, "inertial")
        inertial.set("mass", str(mass))
        inertial.set("diaginertia", "0.0003 0.0003 0.0003")
        inertial.set("pos", "0 0 0")

        # Add a single visible orange sphere geom (group=1 is shown in your viewer)
        sphere = ET.SubElement(previous_object_body, "geom")
        sphere.set("name", f"{object_name}")
        sphere.set("type", "sphere")
        sphere.set("size", str(size))
        sphere.set("rgba", rgba)
        sphere.set("group", "1")  # group 1-5 are visible; group 0 is hidden in your viewer
        sphere.set("mass", str(mass))
        sphere.set("friction", friction)
        
        # Save the modified XML
        kitchen_xml.save_model(self.scene_xml_path)
        print(f"Replaced {object_name} with orange sphere, size: {size}, color: {rgba}")
        
        # Reload the model with the new object
        print("Reloading model with new object...")
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Rebuild joint mappings after model reload
        self._build_joint_mappings()
        print("Model reloaded with orange")