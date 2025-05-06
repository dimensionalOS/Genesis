#!/usr/bin/env python3
"""
Script to patch the lcm_foxglove_bridge_threaded.py to fix PointCloud2 visualization
"""
import os
import sys
import re
import shutil
from datetime import datetime

# Define path to the bridge script
BRIDGE_SCRIPT_PATH = "/home/yashas/Documents/dimensional/lcm_dimos_msgs/python_lcm_msgs/lcm_foxglove_bridge_threaded.py"

def backup_bridge_script():
    """Create a backup of the original bridge script"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{BRIDGE_SCRIPT_PATH}.backup_{timestamp}"
    shutil.copy2(BRIDGE_SCRIPT_PATH, backup_path)
    print(f"Created backup at: {backup_path}")
    return backup_path

def fix_pointcloud_conversion():
    """Fix the PointCloud2 conversion in the bridge script"""
    
    # Create backup
    backup_path = backup_bridge_script()
    
    # Read the original script
    with open(BRIDGE_SCRIPT_PATH, 'r') as f:
        content = f.read()
    
    # Let's define a mapping from PointField datatype numbers to Foxglove numeric types
    # This will be added to the script
    datatype_mapping_code = """
    # Mapping from PointField datatype values to Foxglove PackedElementFieldNumericType enum values
    POINTFIELD_DATATYPE_TO_FOXGLOVE = {
        1: 1,  # INT8
        2: 2,  # UINT8
        3: 3,  # INT16
        4: 4,  # UINT16
        5: 5,  # INT32
        6: 6,  # UINT32
        7: 7,  # FLOAT32
        8: 8,  # FLOAT64
    }
    
    # Type name to numeric type mapping
    POINTFIELD_TYPE_STRINGS = {
        'int8': 1,
        'uint8': 2,
        'int16': 3,
        'uint16': 4,
        'int32': 5,
        'uint32': 6,
        'float32': 7,
        'float64': 8,
    }
"""
    
    # Find the imports section and add our mapping right after it
    imports_end_match = re.search(r'import.*\n\n', content, re.DOTALL)
    if imports_end_match:
        position = imports_end_match.end()
        content = content[:position] + datatype_mapping_code + content[position:]
    
    # Fix the _convert_pointcloud function
    pointcloud_conversion_pattern = r'def _convert_pointcloud\(.*?\):'
    pointcloud_conversion_match = re.search(pointcloud_conversion_pattern, content, re.DOTALL)
    
    if pointcloud_conversion_match:
        # Look for the problematic code that creates PointElementField with string types
        packed_field_pattern = r"PackedElementField\(name=['\"](.*?)['\"]\s*,\s*offset=(\d+)\s*,\s*type=['\"](.*?)['\"]"
        
        # Replace with version that uses numeric types
        fixed_content = re.sub(
            packed_field_pattern,
            lambda m: f"PackedElementField(name='{m.group(1)}', offset={m.group(2)}, type=POINTFIELD_TYPE_STRINGS.get('{m.group(3)}', 7)",  # default to float32 (7) if unknown
            content
        )
        
        # Also fix the case where it's creating fields from PointCloud2 message fields
        field_conversion_pattern = r"(fields\.append\()PackedElementField\(name=field\.name\s*,\s*offset=field\.offset\s*,\s*type=['\"](.*?)['\"]\)"
        fixed_content = re.sub(
            field_conversion_pattern,
            r"\1PackedElementField(name=field.name, offset=field.offset, type=POINTFIELD_DATATYPE_TO_FOXGLOVE.get(field.datatype, 7))",
            fixed_content
        )
        
        # Also add a fallback for when no fields exist
        no_fields_pattern = r"(if not msg\.fields:.*?)(# Create.+?px, py, pz)(\s+fields = \[\s+?PackedElementField)"
        fixed_content = re.sub(
            no_fields_pattern,
            r"\1\2\3",
            fixed_content,
            flags=re.DOTALL
        )
        
        # Write the modified script
        with open(BRIDGE_SCRIPT_PATH, 'w') as f:
            f.write(fixed_content)
        
        print(f"Successfully patched {BRIDGE_SCRIPT_PATH}")
        print(f"Original script backed up to {backup_path}")
        print("\nNow restart the LCM Foxglove bridge to apply changes.")
        return True
    else:
        print("Error: Could not find _convert_pointcloud function in the bridge script.")
        return False

if __name__ == "__main__":
    if not os.path.exists(BRIDGE_SCRIPT_PATH):
        print(f"Error: Bridge script not found at {BRIDGE_SCRIPT_PATH}")
        print("Please update the BRIDGE_SCRIPT_PATH variable in this script.")
        sys.exit(1)
    
    try:
        success = fix_pointcloud_conversion()
        if success:
            print("\nTo test the fix:")
            print("1. Run alfred_lcm.py to start the simulator")
            print("2. Run depth_to_pointcloud.py to generate point clouds")
            print("3. Restart the LCM Foxglove bridge")
            print("4. Connect to the bridge from Foxglove Studio")
            print("5. Add a 3D panel and subscribe to the /head_cam_pointcloud topic")
    except Exception as e:
        print(f"Error patching bridge script: {e}")
        sys.exit(1)