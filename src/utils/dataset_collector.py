"""
Dataset collection utilities for SO101 teleoperation.
Handles data collection, normalization, and saving in lerobot format.

Parquet File Format:
-------------------
Each episode is saved as a .parquet file containing the following columns:

- action: List[float] - Normalized leader arm joint positions [-100, 100] for arm joints, [0, 100] for gripper
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

- observation.state: List[float] - Normalized follower arm joint positions in same format as action
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

- timestamp: float - Time in seconds since episode start

- frame_index: int - Frame number within current episode (0-based)

- episode_index: int - Episode number (0-based)

- index: int - Global frame index within episode (same as frame_index)

- task_index: int - Task identifier (always 0 for single task)

Normalization:
- Arm joints: MuJoCo control range mapped to [-100, 100]
- Gripper: MuJoCo control range mapped to [0, 100] (0=closed, 100=open)
- Raw servo counts (0-4095) are automatically detected and normalized

Video files are saved alongside in videos/chunk-000/{camera_name}/episode_{episode:06d}.mp4
"""

import os
import time
import numpy as np
import pandas as pd
import imageio
import threading
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import mujoco

# Try to import platform-specific keyboard input
try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False
    try:
        import select
        import sys
        import tty
        import termios
        HAS_UNIX_INPUT = True
    except ImportError:
        HAS_UNIX_INPUT = False


class DatasetCollector:
    """Handles dataset collection during teleoperation."""
    
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
    
    def __init__(self, output_dir: str = "collected_datasets", control_fps: int = 30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset structure
        (self.output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)
        
        # Episode tracking
        self.episode_data = []
        self.episode_index = 0
        self.frame_index = 0
        self.start_time = None
        self.task_description = ""
        
        # Video recording
        self.video_writers = {}
        self.video_renderers = {}
        self.camera_names = []
        self.video_fps = control_fps  # Is set to match teleoperation frequency
        self.video_width = 640
        self.video_height = 480
        
        # Keyboard handling
        self.keyboard_thread = None
        self.should_reset_episode = False
        self.should_quit = False
        self.should_toggle_recording = False
        self.should_save_episode = False
        self.should_discard_episode = False
        self.waiting_for_save_decision = False
        
        # Collection state
        self.is_collecting = False
        
        # Episode tracking for meta files
        self.episodes_info = []  # List of episode metadata
        self.total_frames_collected = 0
        
        # Recording timeout
        self.recording_timeout = 10  # Default timeout in seconds
        
        self.camera_name_mapping = {
            "overhead_camera": "top",
            "wrist_camera": "wrist"
        }
    
    def start_collection(self, task_description: str = ""):
        """Start dataset collection."""
        self.task_description = task_description
        
        # Only reset episode_index if this is the very first episode
        if self.start_time is None and self.episode_index == 0:
            self.episode_data = []
            self.frame_index = 0
        
        self.start_time = time.time()
        self.is_collecting = True
        
        print(f"Started dataset collection in {self.output_dir}")
        print(f"Task: {self.task_description}")
    
    def setup_video_recording(self, model: mujoco.MjModel, camera_names: List[str], 
                            width: int = 640, height: int = 480, fps: int = 30):
        """Setup video recording for the specified cameras."""
        self.camera_names = camera_names
        self.video_width = width
        self.video_height = height
        self.video_fps = fps
        
        # Create video directories
        for cam_name in camera_names:
            video_key = self.camera_name_mapping.get(cam_name, cam_name)
            video_key = f"observation.images.{video_key}"
            cam_dir = self.output_dir / "videos" / "chunk-000" / video_key
            cam_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize renderers
        for cam_name in camera_names:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id == -1:
                print(f"Warning: camera '{cam_name}' not found. Skipping.")
                continue
            
            self.video_renderers[cam_name] = mujoco.Renderer(model, width=width, height=height)
    
    def _normalize_leader_action(self, leader_action: Dict[str, Any]) -> np.ndarray:
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
    
    def _normalize_follower_positions(self, follower_positions: List[float]) -> List[float]:
        """Normalize follower joint positions to [-100, 100] range."""
        normalized_positions = []
        
        for i, joint_name in enumerate(self.JOINT_ORDER):
            ctrl_min, ctrl_max = self.CTRL_RANGES[joint_name]
            pos = follower_positions[i]
            
            if joint_name == "gripper":
                # Gripper: [ctrl_min, ctrl_max] to [0, 100]
                normalized = (pos - ctrl_min) / (ctrl_max - ctrl_min) * 100.0
            else:
                # Arm joints: [ctrl_min, ctrl_max] to [-100, 100]
                normalized = ((pos - ctrl_min) / (ctrl_max - ctrl_min) - 0.5) * 200.0
            
            normalized_positions.append(normalized)
        
        return normalized_positions
    
    def collect_frame(self, leader_action: Dict[str, Any], follower_positions: List[float], 
                     data: mujoco.MjData, offscreen_scene_option: mujoco.MjvOption):
        """Collect data for current frame."""
        if not self.is_collecting:
            return
        
        # Get current timestamp
        current_time = time.time()
        timestamp = current_time - self.start_time
        
        # Get normalized leader action (actions)
        normalized_leader_action = self._normalize_leader_action(leader_action)
        
        # Get normalized follower positions (observation.state)
        normalized_follower_positions = self._normalize_follower_positions(follower_positions)
        
        # Create frame data
        frame_data = {
            "action": normalized_leader_action.tolist(),
            "observation.state": normalized_follower_positions,
            "timestamp": timestamp,
            "frame_index": self.frame_index,
            "episode_index": self.episode_index,
            "index": len(self.episode_data),
            "task_index": 0  # Single task for now
        }
        
        self.episode_data.append(frame_data)
        self.frame_index += 1
        
        # Record video frames
        self._record_video_frames(data, offscreen_scene_option)
    
    def _record_video_frames(self, data: mujoco.MjData, offscreen_scene_option: mujoco.MjvOption):
        """Record video frames for all cameras."""
        for cam_name in self.camera_names:
            if cam_name not in self.video_renderers:
                continue
            
            renderer = self.video_renderers[cam_name]
            
            # Render with group 0 hidden (collision meshes)
            renderer.update_scene(data, camera=cam_name, scene_option=offscreen_scene_option)
            rgb = renderer.render()
            
            # Convert to uint8
            if rgb.dtype != np.uint8:
                frame = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                frame = rgb
            
            # Get or create video writer for this camera and episode
            writer_key = f"{cam_name}_{self.episode_index}"
            if writer_key not in self.video_writers:
                video_key = self.camera_name_mapping.get(cam_name, cam_name)
                video_key = f"observation.images.{video_key}"

                video_path = self.output_dir / "videos" / "chunk-000" / video_key / f"episode_{self.episode_index:06d}.mp4"
                try:
                    self.video_writers[writer_key] = imageio.get_writer(str(video_path), fps=self.video_fps)
                    print(f"Started recording {video_key} to {video_path}")
                except Exception as e:
                    print(f"Warning: could not open MP4 writer for {video_key} ({e}). Skipping video recording.")
                    self.video_writers[writer_key] = None
            
            # Write frame
            writer = self.video_writers[writer_key]
            if writer is not None:
                writer.append_data(frame)
    
    def save_episode(self):
        """Save current episode data to parquet file."""
        if not self.episode_data:
            print("No data to save for current episode")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.episode_data)
        
        # Save to parquet
        parquet_path = self.output_dir / "data" / "chunk-000" / f"episode_{self.episode_index:06d}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        print(f"✅ Saved episode {self.episode_index} with {len(self.episode_data)} frames to {parquet_path}")
        
        # Store episode metadata
        episode_meta = {
            "episode_index": self.episode_index,
            "tasks": [self.task_description],
            "length": len(self.episode_data)
        }
        self.episodes_info.append(episode_meta)
        self.total_frames_collected += len(self.episode_data)
        
        # Close video writers for this episode
        for writer_key in list(self.video_writers.keys()):
            if writer_key.endswith(f"_{self.episode_index}"):
                writer = self.video_writers[writer_key]
                if writer is not None:
                    try:
                        writer.close()
                        print(f"Closed video writer for {writer_key}")
                    except Exception as e:
                        print(f"Warning: Error closing video writer {writer_key}: {e}")
                del self.video_writers[writer_key]
        
        # Prepare for next episode
        self.prepare_next_episode()

    def discard_current_episode_data(self):
        """Discard current episode data without saving."""
        if not self.episode_data:
            print("No data to discard for current episode")
            return
            
        print(f"🗑️ Discarded episode {self.episode_index} with {len(self.episode_data)} frames")
        
        
        
        # Close and remove video files for this episode without saving metadata
        for writer_key in list(self.video_writers.keys()):
            if writer_key.endswith(f"_{self.episode_index}"):
                writer = self.video_writers[writer_key]
                if writer is not None:
                    try:
                        writer.close()
                        print(f"Closed video writer for {writer_key}")
                    except Exception as e:
                        print(f"Warning: Error closing video writer {writer_key}: {e}")
                del self.video_writers[writer_key]
        
        # Remove video files for this episode
        for cam_name in self.camera_names:
            video_key = self.camera_name_mapping.get(cam_name, cam_name)
            video_key = f"observation.images.{video_key}"
            video_path = self.output_dir / "videos" / "chunk-000" / video_key / f"episode_{self.episode_index:06d}.mp4"
            if video_path.exists():
                try:
                    video_path.unlink()
                    print(f"Removed video file: {video_path}")
                except Exception as e:
                    print(f"Warning: Could not remove video file {video_path}: {e}")
        
        # Remove parquet file if it exists (shouldn't exist yet, but just in case)
        parquet_path = self.output_dir / "data" / "chunk-000" / f"episode_{self.episode_index:06d}.parquet"
        if parquet_path.exists():
            try:
                parquet_path.unlink()
                print(f"Removed parquet file: {parquet_path}")
            except Exception as e:
                print(f"Warning: Could not remove parquet file {parquet_path}: {e}")
        
        # Prepare for next episode (reuse same episode index since we're discarding)
        self.episode_data = []
        self.frame_index = 0
        self.start_time = None
        self.is_collecting = False
        # Note: episode_index stays the same since we're discarding this episode
        
        print(f"Ready to record episode {self.episode_index} again or continue with next episode")
    
    def prepare_next_episode(self):
        """Prepare for the next episode recording."""
        self.episode_data = []
        self.episode_index += 1
        self.frame_index = 0
        self.start_time = None  # Will be set when collection starts
        self.is_collecting = False
    
    def discard_current_episode(self):
        """Discard current episode data without saving."""
        if self.episode_data:
            print(f"Discarding episode {self.episode_index} with {len(self.episode_data)} frames")
            
            # Close video writers for this episode without saving
            for writer_key in list(self.video_writers.keys()):
                if writer_key.endswith(f"_{self.episode_index}"):
                    writer = self.video_writers[writer_key]
                    if writer is not None:
                        try:
                            writer.close()
                        except Exception:
                            pass
                    del self.video_writers[writer_key]
        
        # Reset for new episode
        self.episode_data = []
        self.frame_index = 0
        self.start_time = time.time()
        self.is_collecting = False
        
        print(f"Episode {self.episode_index} discarded, ready for new recording")
    
    def _keyboard_input_handler(self):
        """Handle keyboard input in a separate thread."""
        print("Keyboard handler started. Press S to start/stop recording, R to reset, T to save, D to discard, Q to quit.")
        
        if not HAS_MSVCRT and not HAS_UNIX_INPUT:
            print("Warning: No keyboard input available. Use Ctrl+C to quit.")
            while not self.should_quit:
                time.sleep(1)
            return
        
        while not self.should_quit:
            try:
                key = None
                
                if HAS_MSVCRT:
                    # Windows keyboard input
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8').upper()
                elif HAS_UNIX_INPUT:
                    # Unix/Linux keyboard input (non-blocking)
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).upper()
                
                if key:
                    print(f"Key pressed: {key}")  # Debug print
                    
                    if self.waiting_for_save_decision:
                        # Handle save/discard decision
                        if key == 'T':
                            self.should_save_episode = True
                            self.waiting_for_save_decision = False
                            print("\n💾 Save episode requested...")
                        elif key == 'D':
                            self.should_discard_episode = True
                            self.waiting_for_save_decision = False
                            print("\n🗑️ Discard episode requested...")
                        else:
                            print(f"\nInvalid key '{key}'. Press 'T' to save or 'D' to discard the episode.")
                    else:
                        # Handle normal commands
                        if key == 'R':
                            self.should_reset_episode = True
                            print("\n🔄 Environment reset requested...")
                        elif key == 'S':
                            self.should_toggle_recording = True
                            print("\n📹 Recording toggle requested...")
                        elif key == 'Q':
                            self.should_quit = True
                            print("\n❌ Quit requested...")
                            break
                        
            except Exception as e:
                print(f"Keyboard handler error: {e}")  # Debug print
            
            time.sleep(0.1)  # Small delay to prevent high CPU usage
        print("Keyboard handler stopped.")
    
    def start_keyboard_handler(self):
        """Start keyboard input handler thread."""
        if self.keyboard_thread is None:
            self.keyboard_thread = threading.Thread(target=self._keyboard_input_handler, daemon=True)
            self.keyboard_thread.start()
    
    def stop_keyboard_handler(self):
        """Stop keyboard input handler thread."""
        print("Stopping keyboard handler...")
        self.should_quit = True
        if self.keyboard_thread and self.keyboard_thread.is_alive():
            self.keyboard_thread.join(timeout=2.0)
            if self.keyboard_thread.is_alive():
                print("Warning: Keyboard handler thread did not stop gracefully")
        print("Keyboard handler stopped")
    
    def stop_collection(self):
        """Stop dataset collection and save final episode."""
        if not self.is_collecting:
            return
        
        # Save final episode if there's data
        if self.episode_data:
            self.save_episode()
        
        # Close all video writers
        for writer in self.video_writers.values():
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
        
        # Close renderers
        for renderer in self.video_renderers.values():
            try:
                renderer.close()
            except Exception:
                pass
        
        self.is_collecting = False
        print("Dataset collection stopped")
    
    def generate_meta_files(self):
        """Generate all meta files for the dataset."""
        meta_dir = self.output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating meta files in {meta_dir}...")
        
        # Generate info.json
        self._generate_info_json(meta_dir)
        
        # Generate episodes.jsonl
        self._generate_episodes_jsonl(meta_dir)
        
        # Generate tasks.jsonl
        self._generate_tasks_jsonl(meta_dir)
        
        # Generate episodes_stats.jsonl
        self._generate_episodes_stats_jsonl(meta_dir)
        
        print("Meta files generated successfully!")
    
    def _generate_info_json(self, meta_dir: Path):
        """Generate info.json with dataset metadata."""
        total_episodes = len(self.episodes_info)
        total_videos = total_episodes * len(self.camera_names)
        
        # Map camera names to video keys
        
        
        info = {
            "codebase_version": "v2.1",
            "robot_type": "so101", 
            "total_episodes": total_episodes,
            "total_frames": self.total_frames_collected,
            "total_tasks": 1,
            "total_videos": total_videos,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.video_fps,
            "splits": {
                "train": f"0:{total_episodes}"
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": [
                        "main_shoulder_pan",
                        "main_shoulder_lift", 
                        "main_elbow_flex",
                        "main_wrist_flex",
                        "main_wrist_roll",
                        "main_gripper"
                    ]
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": [
                        "main_shoulder_pan",
                        "main_shoulder_lift",
                        "main_elbow_flex", 
                        "main_wrist_flex",
                        "main_wrist_roll",
                        "main_gripper"
                    ]
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "episode_index": {
                    "dtype": "int64", 
                    "shape": [1],
                    "names": None
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1], 
                    "names": None
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                }
            }
        }
        
        # Add video features for each camera
        for cam_name in self.camera_names:
            video_key = self.camera_name_mapping.get(cam_name, cam_name)
            info["features"][f"observation.images.{video_key}"] = {
                "dtype": "video",
                "shape": [self.video_height, self.video_width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": float(self.video_fps),
                    "video.height": self.video_height,
                    "video.width": self.video_width,
                    "video.channels": 3,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p", 
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            }
        
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4)
        print(f"Generated info.json with {total_episodes} episodes")
    
    def _generate_episodes_jsonl(self, meta_dir: Path):
        """Generate episodes.jsonl with episode metadata."""
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for episode_info in self.episodes_info:
                f.write(json.dumps(episode_info) + "\n")
        print(f"Generated episodes.jsonl with {len(self.episodes_info)} episodes")
    
    def _generate_tasks_jsonl(self, meta_dir: Path):
        """Generate tasks.jsonl with task definitions."""
        with open(meta_dir / "tasks.jsonl", "w") as f:
            task_entry = {
                "task_index": 0,
                "task": self.task_description
            }
            f.write(json.dumps(task_entry) + "\n")
        print("Generated tasks.jsonl")
    
    def _generate_episodes_stats_jsonl(self, meta_dir: Path):
        """Generate episodes_stats.jsonl with statistical information for each episode."""
        print("Calculating episode statistics...")
        
        with open(meta_dir / "episodes_stats.jsonl", "w") as f:
            for episode_info in self.episodes_info:
                episode_idx = episode_info["episode_index"]
                print(f"Processing statistics for episode {episode_idx}...")
                
                # Load the episode data to calculate statistics
                parquet_path = self.output_dir / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
                
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                    
                    # Calculate statistics for action and observation.state
                    stats = {
                        "episode_index": episode_idx,
                        "stats": {}
                    }
                    
                    for column in ["action", "observation.state"]:
                        if column in df.columns:
                            # Convert list column to numpy array for statistics
                            data_array = np.array(df[column].tolist())
                            
                            stats["stats"][column] = {
                                "min": data_array.min(axis=0).tolist(),
                                "max": data_array.max(axis=0).tolist(),
                                "mean": data_array.mean(axis=0).tolist(),
                                "std": data_array.std(axis=0).tolist(),
                                "count": [data_array.shape[0]],
                            }
                    
                    # Add other scalar statistics
                    for column in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
                        if column in df.columns:
                            stats["stats"][column] = {
                                "min": [float(df[column].min())],
                                "max": [float(df[column].max())],
                                "mean": [float(df[column].mean())],
                                "std": [float(df[column].std())],
                                "count": [df[column].shape[0]],
                            }
                    
                    # Add video statistics
                    print(f"Processing video statistics for episode {episode_idx}...")
                    video_stats = self._process_video_stats(episode_idx, self.output_dir / "videos" / "chunk-000")
                    stats["stats"].update(video_stats)
                    
                    
                    f.write(json.dumps(stats) + "\n")
                    
                else:
                    print(f"Warning: Episode file {parquet_path} not found for statistics")
        
        print(f"Generated episodes_stats.jsonl with statistics for {len(self.episodes_info)} episodes")

    def _process_video_stats(self, episode_idx, video_dir: Path):
        
        """Process videos and extract the video-related statistics for two camera views of the same episode"""
        
        camera_name_mapping = {
            "overhead_camera": "top",
            "wrist_camera": "wrist"
        }
        
        stats = {}
        
        for cam_name in self.camera_names:
            video_key = camera_name_mapping.get(cam_name, cam_name)
            video_key = f"observation.images.{video_key}"
            
            # Process the video for this camera view of the episode
            video_path = video_dir / video_key / f"episode_{episode_idx:06d}.mp4"
            
            if not video_path.exists():
                print(f"Warning: Video file {video_path} not found for episode {episode_idx}")
                continue
            
            try:
                video = imageio.get_reader(video_path)
                frames = [frame for frame in video]
                video.close()  # Close the video reader
                
                if not frames:
                    print(f"Warning: No frames found in video {video_path}")
                    continue
                
                frames = np.array(frames)
                frames = frames.reshape(-1, self.video_height, self.video_width, 3)
                # Ensure frames are in uint8 range [0, 255] before normalization
                frames = np.clip(frames, 0, 255).astype(np.uint8)
                # Normalize by dividing by 255 (convert to float32 range [0, 1])
                frames = frames.astype(np.float32) / 255.0
                
                # Sample every 3rd frame for statistics
                sampled_frames = frames[::3]
                
                # Calculate statistics for each channel (R, G, B)
                min_vals = []
                max_vals = []
                mean_vals = []
                std_vals = []
                
                for channel in range(3):  # RGB channels
                    channel_data = sampled_frames[:, :, :, channel]
                    min_vals.append([[float(channel_data.min())]])
                    max_vals.append([[float(channel_data.max())]])
                    mean_vals.append([[float(channel_data.mean())]])
                    std_vals.append([[float(channel_data.std())]])
                
                stats[video_key] = {
                    "min": min_vals,
                    "max": max_vals,
                    "mean": mean_vals,
                    "std": std_vals,
                    "count": [len(sampled_frames)],
                }
                
                print(f"Processed video stats for {video_key}: {len(sampled_frames)} frames")
                
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                continue
            
        print(f"Processed video stats for episode {episode_idx}: {stats}")
        return stats
    
    
    
    def cleanup(self):
        """Clean up all resources and generate meta files."""
        print("Starting dataset collector cleanup...")
        
        # Stop keyboard handler first
        self.stop_keyboard_handler()
        
        # Generate meta files before cleanup
        if self.episodes_info:  # Only if we have collected episodes
            print(f"Generating meta files for {len(self.episodes_info)} episodes...")
            try:
                self.generate_meta_files()
                print("Meta files generated successfully!")
            except Exception as e:
                print(f"Error generating meta files: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No episodes collected - skipping meta file generation")
        
        # Stop collection and close all resources
        self.stop_collection()
        
        print("Dataset collector cleanup completed")
