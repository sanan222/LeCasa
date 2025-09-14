"""
Interactive CLI entrypoint to build a RoboCasa kitchen and run a selected task.

Behavior (per README):
- Prompt (or accept CLI args) for layout, style, and task
- Create the kitchen using our task wrappers under `src/tasks/`
- Optionally visualize the environment (static Mujoco viewer)

Note:
- Teleoperation wiring can be layered on top (see `tasks/teleoperator.py`).
"""

from __future__ import annotations

import argparse
import os
import sys
import signal
from typing import Optional

# os.environ.setdefault("NV_OPTIMUS_ENABLE", "1")


"""Set the MuJoCo GL backend before importing any modules that might import mujoco."""
if os.name == "nt":  # Windows
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

from tasks.teleoperator import SO101TeleoperationController
from tasks.single_stage_so101.tasks.banana_to_bowl import BananaToBowlTeleop
from tasks.single_stage_so101.tasks.orange_to_plate import OrangeToPlateTeleop
from tasks.single_stage_so101.tasks.orange_to_bowl import OrangeToBowlTeleop
from tasks.single_stage_so101.tasks.turn_on_microwave import TurnOnMicrowaveTeleop
from tasks.single_stage_so101.tasks.turn_on_stove import TurnOnStoveTeleop
from tasks.single_stage_so101.tasks.microwave_door import MicrowaveDoorTeleop
from tasks.single_stage_so101.tasks.drawer import DrawerTeleop
from tasks.single_stage_so101.tasks.pan_to_sink import PanToSinkTeleop


from tasks.multi_stage_so101.tasks.fruitbowl import FruitBowlTeleop


# Ensure this script can import local modules when run directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


# (moved above, before any potential mujoco import)


# Optional dependencies
try:  # MuJoCo may be missing on some dev machines
    import mujoco  # type: ignore
    MUJOCO_AVAILABLE = True
except Exception:
    MUJOCO_AVAILABLE = False

try:
    from termcolor import colored
except Exception:  # Fallback if termcolor unavailable
    def colored(text: str, _color: str) -> str:  # type: ignore
        return text


# Local options (avoid importing heavy demo helpers with external deps)
LAYOUT_OPTIONS = [
    "One Wall Small", "One Wall Large", "L-Shaped Small", "L-Shaped Large",
    "U-Shaped Small", "U-Shaped Large", "Wraparound", "Island",
    "Peninsula", "Galley",
]

STYLE_OPTIONS = [
    "Modern", "Traditional", "Contemporary", "Rustic", "Industrial",
    "Minimalist", "Scandinavian", "Mediterranean", "Asian", "Coastal",
    "Farmhouse", "Urban",
]


TASK_NAMES = [
    "BananaToBowlPnP",
    "OrangeToPlatePnP",
    "OrangeToBowlPnP",
    "PanToSinkPnP",
    "TurnOnMicrowave",
    "TurnOnStove",
    "MicrowaveDoor",
    "Drawer",
    "FruitBowlPnP",
]

TASK_DESCRIPTIONS = {
    "BananaToBowlPnP": "pick a banana from counter and place it into a bowl",
    "OrangeToPlatePnP": "pick an orange from counter and place it on a plate",
    "OrangeToBowlPnP": "pick an orange from counter and place it into a bowl",
    "PanToSinkPnP": "pick a pan from counter and place it into a sponge",
    "TurnOnMicrowave": "press the start button on the microwave",
    "TurnOnStove": "turn on the stove burner",
    "MicrowaveDoor": "open/close the microwave door",
    "Drawer": "open/close a kitchen drawer",
    "FruitBowlPnP": "make a fruit bowl",
}


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    print(f"\nReceived signal {signum} - initiating graceful shutdown...")
    sys.exit(0)

def main():
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="SO101 RoboCasa interactive kitchen CLI")
    parser.add_argument("--task", type=str, help=f"Task name ({', '.join(TASK_NAMES)})")
    parser.add_argument("--layout", type=int, help="Kitchen layout ID (0-9)")
    parser.add_argument("--style", type=int, help="Kitchen style ID (0-11)")
    parser.add_argument("--random-env", action="store_true", help="Randomize layout and style")
    parser.add_argument("--port", type=str, default=os.environ.get("SO101_PORT", "COM5"), help="Serial port for leader arm (default: COM5 or $SO101_PORT)")
    parser.add_argument("--calibration", type=str, default=None, help="Path to calibration JSON under configs/robot_configs (e.g., configs/robot_configs/my_leader.json)")
    parser.add_argument("--no-render", action="store_true", help="Do not launch MuJoCo viewer")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")
    
    # Dataset collection options
    parser.add_argument("--dataset-output-dir", type=str, default="collected_datasets", help="Directory to save dataset files (default: collected_datasets)")
    parser.add_argument("--recording-timeout", type=int, default=10, help="Maximum recording time in seconds before auto-stop (default: 10)")
    parser.add_argument("--fps", type=int, default=30, help="FPS for control and video recording (default: 30)")
    args = parser.parse_args()

    if args.list_tasks:
        print("Available tasks:")
        for name in TASK_NAMES:
            print(f"  - {name}")
        return

    # If any of task/layout/style are missing, show all options together with quick explanations
    task_name: Optional[str] = args.task
    layout_id: Optional[int] = args.layout
    style_id: Optional[int] = args.style

    if task_name is None or layout_id is None or style_id is None:
        task_names = TASK_NAMES
        layout_options = LAYOUT_OPTIONS
        style_options = STYLE_OPTIONS

        print("Tasks:")
        for i, name in enumerate(task_names):
            desc = TASK_DESCRIPTIONS.get(name, "")
            print(f"  [{i}] {name}: {desc}")
        print()

        print("Kitchen layouts:")
        for i, label in enumerate(layout_options):
            print(f"  [{i}] {label}")
        print()

        print("Kitchen styles:")
        for i, label in enumerate(style_options):
            print(f"  [{i}] {label}")
        print()

        # Prompt for any missing values
        if task_name is None:
            try:
                s = input(f"Select task 0-{len(task_names) - 1} (default 0 - {task_names[0]}): ")
                k = min(max(int(s), 0), len(task_names) - 1)
            except Exception:
                k = 0
                print(f"Use {task_names[k]} by default.\n")
            task_name = task_names[k]

        if layout_id is None:
            try:
                s = input("Select layout 0-{} (default 0): ".format(len(layout_options) - 1))
                layout_id = int(s)
                if layout_id < 0:
                    layout_id = 0
                if layout_id > len(layout_options) - 1:
                    layout_id = len(layout_options) - 1
            except Exception:
                layout_id = 0
                print("Use layout 0 by default.\n")

        if style_id is None:
            try:
                s = input("Select style 0-{} (default 0): ".format(len(style_options) - 1))
                style_id = int(s)
                if style_id < 0:
                    style_id = 0
                if style_id > len(style_options) - 1:
                    style_id = len(style_options) - 1
            except Exception:
                style_id = 0
                print("Use style 0 by default.\n")

    # Announce selection
    layout_options = LAYOUT_OPTIONS
    style_options = STYLE_OPTIONS
    print(colored(f"\nSelected task: {task_name}", "yellow"))
    print(f"Layout: {layout_id} ({layout_options[layout_id]})")
    print(f"Style: {style_id} ({style_options[style_id]})")
    
    # Announce dataset collection settings
    print(colored(f"Dataset collection: KEYBOARD CONTROLLED", "cyan"))
    print(f"  Output directory: {args.dataset_output_dir}")
    print(f"  Recording timeout: {args.recording_timeout} seconds")
    print("  Press 'S' during teleoperation to start/stop recording")

    # Resolve calibration path under repo configs if not provided
    if args.calibration is None:
        default_calib = os.path.normpath(os.path.join(THIS_DIR, "..", "configs", "robot_configs", "my_leader.json"))
        calib_path = default_calib
        # If default missing, try to pick the first .json in configs/robot_configs
        calib_dir = os.path.dirname(default_calib)
        try:
            if not os.path.exists(calib_path) and os.path.isdir(calib_dir):
                for name in os.listdir(calib_dir):
                    if name.lower().endswith(".json"):
                        calib_path = os.path.join(calib_dir, name)
                        break
        except Exception:
            pass
    else:
        calib_path = args.calibration

    # Run teleoperation using the unified controller
    if not args.no_render:
        if MUJOCO_AVAILABLE:
            robot_xml_path = os.path.join(THIS_DIR, "robot", "so101_simplified.xml")
            # Instantiate task-specific teleop controller
            teleop_cls = {
                "BananaToBowlPnP": BananaToBowlTeleop,
                "OrangeToPlatePnP": OrangeToPlateTeleop,
                "OrangeToBowlPnP": OrangeToBowlTeleop,
                "PanToSinkPnP": PanToSinkTeleop,
                "TurnOnMicrowave": TurnOnMicrowaveTeleop,
                "TurnOnStove": TurnOnStoveTeleop,
                "MicrowaveDoor": MicrowaveDoorTeleop,
                "Drawer": DrawerTeleop,
                "FruitBowlPnP": FruitBowlTeleop,
            }[task_name]
            controller = teleop_cls(
                port=args.port,
                calibration_file=calib_path,
                robot_xml_path=robot_xml_path,
                layout_id=layout_id,
                style_id=style_id,
                task_name=task_name,
                fps=args.fps
            )
            
            try:
                controller.initialize_hardware()
                controller.initialize_kitchen_environment()
                controller.integrate_robot_into_kitchen()
                controller.initialize_mujoco_simulation()
                controller._replace_object_with_object("banana_joint0", size=0.03, mass=0.2, friction="1 0.5 0.1", rgba="1 0.5 0 1")
                
                # Run teleoperation with keyboard-controlled dataset collection
                controller.run_teleoperation(
                    dataset_output_dir=args.dataset_output_dir,
                    recording_timeout=args.recording_timeout
                )
            except KeyboardInterrupt:
                print("\nReceived Ctrl+C - shutting down gracefully...")
            except Exception as e:
                print(f"Error during teleoperation: {e}")
            finally:
                print("Performing final cleanup and shutdown...")
                controller.cleanup()
                print("All systems shut down. Exiting program.")
                import sys
                sys.exit(0)
        else:
            print(colored("MuJoCo not available. Skipping teleoperation.", "red"))


if __name__ == "__main__":
    main()


