"""
SO101 Leader Hardware Interface

Provides real-time position reading from SO101 leader arm using lerobot framework.
Handles calibration loading and hardware communication.
"""

import time
import os
import sys
import json
from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path

# Add lerobot_oft to Python path
lerobot_path = os.path.abspath("../lerobot_oft/src")
if lerobot_path not in sys.path:
    sys.path.insert(0, lerobot_path)

try:
    from lerobot.teleoperators.so101_leader import SO101Leader as LeRobotSO101Leader
    from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig as LeRobotConfig
    LEROBOT_AVAILABLE = True
except ImportError as e:
    LEROBOT_AVAILABLE = False
    print(f"LeRobot import failed: {e}")
    print("   Ensure you're in the 'lerobot' conda environment")


@dataclass
class SO101LeaderConfig:
    """Configuration for SO101 Leader"""
    port: str = "COM5"
    use_degrees: bool = False
    calibration_file: str = "my_leader.json"
    calibration_dir: Optional[str] = None


class SO101Leader:
    """
    Real SO101 Leader interface using user's conda environment and calibration.
    """
    
    JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    def __init__(self, config: SO101LeaderConfig):
        self.config = config
        self.is_connected = False
        self._hardware_leader = None
        self._last_positions = {joint: 0.0 for joint in self.JOINT_ORDER}
        
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot components not available. Make sure you're in the 'lerobot' conda environment")
        
        # Set up calibration path
        calibration_dir = config.calibration_dir or os.getcwd()
        calibration_path = os.path.join(calibration_dir, config.calibration_file)
        
        if not os.path.exists(calibration_path):
            print(f"Calibration file not found at: {calibration_path}")
            print(f"   Looking for calibration files in current directory...")
            self._find_calibration_files()
        else:
            print(f"Found calibration file: {calibration_path}")
        
        # Create lerobot configuration
        try:
            lerobot_config = LeRobotConfig(
                port=config.port,
                use_degrees=config.use_degrees,
                id="my_leader"  # This should match the calibration file name (without .json)
            )
            
            # Override calibration directory if specified
            if config.calibration_dir:
                lerobot_config.calibration_dir = Path(config.calibration_dir)
            
            self._hardware_leader = LeRobotSO101Leader(lerobot_config)
            print(f"Hardware leader interface created for port {config.port}")
            
        except Exception as e:
            print(f"Failed to create hardware interface: {e}")
            raise
    
    def _find_calibration_files(self):
        """Find available calibration files"""
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        if json_files:
            print("   Available calibration files:")
            for f in json_files:
                print(f"     - {f}")
        else:
            print("   No .json calibration files found in current directory")
    
    def connect(self, calibrate: bool = False) -> None:
        """Connect to the leader arm"""
        if self.is_connected:
            print("SO101 Leader already connected")
            return
        
        try:
            print(f"Connecting to SO101 leader on {self.config.port}...")
            
            # Connect to hardware (skip calibration since user has calibration file)
            self._hardware_leader.connect(calibrate=calibrate)
            
            # Test reading initial position
            initial_action = self._hardware_leader.get_action()
            
            # Update internal tracking
            self._update_positions_from_action(initial_action)
            
            self.is_connected = True
            print("Hardware leader connected successfully!")
            
        except Exception as e:
            print(f"Hardware connection failed: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from the leader arm"""
        if not self.is_connected:
            return
        
        try:
            if self._hardware_leader:
                self._hardware_leader.disconnect()
            print("Hardware leader disconnected")
        except Exception as e:
            print(f"Disconnect error: {e}")
        
        self.is_connected = False
    
    def get_action(self) -> Dict[str, float]:
        """Get current joint positions in lerobot format"""
        if not self.is_connected or not self._hardware_leader:
            return {f"{joint}.pos": self._last_positions[joint] for joint in self.JOINT_ORDER}
        
        try:
            action = self._hardware_leader.get_action()
            self._update_positions_from_action(action)
            return action
        except Exception as e:
            print(f"Error reading from hardware: {e}")
            # Return last known positions
            return {f"{joint}.pos": self._last_positions[joint] for joint in self.JOINT_ORDER}
    
    def get_joint_positions(self) -> Dict[str, float]:
        """Get joint positions as simple dictionary"""
        return self._last_positions.copy()
    
    def _update_positions_from_action(self, action: Dict[str, float]):
        """Update internal position tracking from lerobot action format"""
        for joint in self.JOINT_ORDER:
            key = f"{joint}.pos"
            if key in action:
                self._last_positions[joint] = action[key]


def create_leader_config(port: str = "COM5", calibration_file: str = "my_leader.json", calibration_dir: Optional[str] = None) -> SO101LeaderConfig:
    """Helper function to create leader configuration"""
    return SO101LeaderConfig(
        port=port,
        use_degrees=False,
        calibration_file=calibration_file,
        calibration_dir=calibration_dir
    )


def test_hardware_connection(port: str = "COM5", calibration_file: str = "my_leader.json"):
    """Test function to verify hardware connection with user's calibration"""
    print(f"Testing hardware connection...")
    print(f"   Port: {port}")
    print(f"   Calibration: {calibration_file}")
    
    try:
        config = create_leader_config(port=port, calibration_file=calibration_file)
        leader = SO101Leader(config)
        
        leader.connect(calibrate=False)  # Use existing calibration
        
        print("Hardware connection successful!")
        print("Move your leader arm to see position changes...")
        
        # Test reading positions
        last_positions = None
        for i in range(20):  # 10 seconds at 2 Hz
            try:
                action = leader.get_action()
                positions = leader.get_joint_positions()
                
                # Print when significant change detected
                if last_positions is None or any(
                    abs(positions[joint] - last_positions[joint]) > 3.0 
                    for joint in positions
                ):
                    print(f"   Action: {action}")
                    print(f"   Positions: {positions}")
                    print()
                    last_positions = positions.copy()
                    
                time.sleep(0.5)  # 2 Hz
                
            except KeyboardInterrupt:
                print("   Test interrupted by user")
                break
                
        leader.disconnect()
        print("Hardware test completed!")
        return True
        
    except Exception as e:
        print(f"Hardware test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test real SO101 Leader hardware")
    parser.add_argument('--port', default='COM5', help='Serial port for leader arm')
    parser.add_argument('--calibration', default='my_leader.json', help='Calibration file name')
    parser.add_argument('--test', action='store_true', help='Run connection test')
    
    args = parser.parse_args()
    
    print("SO101 Leader Hardware Interface")
    print("===============================")
    print("Make sure you're running in the 'lerobot' conda environment!")
    print()
    
    if args.test:
        success = test_hardware_connection(args.port, args.calibration)
        if success:
            print("Your hardware is ready for teleoperation!")
        else:
            print("Hardware test failed - check your setup")
    else:
        # Quick connection test
        try:
            config = create_leader_config(port=args.port, calibration_file=args.calibration)
            leader = SO101Leader(config)
            print("Hardware interface created successfully")
            print("   Use --test flag to run a full connection test")
        except Exception as e:
            print(f"Failed to create hardware interface: {e}") 