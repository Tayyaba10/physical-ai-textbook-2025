---
title: Ch12 - Isaac Sim – Photorealistic Simulation & Synthetic Data
module: 3
chapter: 12
sidebar_label: Ch12: Isaac Sim – Photorealistic Simulation & Synthetic Data
description: Creating photorealistic simulations and generating synthetic data with Isaac Sim
tags: [isaac-sim, omniverse, simulation, synthetic-data, photorealistic, rendering, ai-training]
difficulty: advanced
estimated_duration: 120
---

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Isaac Sim – Photorealistic Simulation & Synthetic Data

## Learning Outcomes
- Create photorealistic simulation environments using Isaac Sim and Omniverse
- Generate synthetic image and sensor data for AI training
- Understand the USD (Universal Scene Description) format and workflows
- Configure advanced rendering features including RTX ray tracing
- Implement domain randomization techniques for robust AI models
- Validate synthetic data quality and realism
- Integrate synthetic data generation into AI development workflows

## Theory

### Isaac Sim and Omniverse Integration

Isaac Sim is built on NVIDIA's Omniverse platform, which provides a collaborative environment for 3D design workflows. This integration provides:

<MermaidDiagram chart={`
graph TD;
    A[Isaac Sim] --> B[Omniverse Kit];
    A --> C[PhysX Physics];
    A --> D[RTX Rendering];
    A --> E[USD Format];
    
    B --> F[Extensibility Framework];
    B --> G[Multi-App Collaboration];
    B --> H[Python API];
    
    C --> I[Realistic Physics Simulation];
    C --> J[Collision Detection];
    C --> K[Rigid Body Dynamics];
    
    D --> L[Ray Tracing];
    D --> M[Global Illumination];
    D --> N[Physically-based Materials];
    
    E --> O[Scalable Scene Representation];
    E --> P[Asset Interchange];
    E --> Q[Animation Support];
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style D fill:#2196F3,stroke:#0D47A1,color:#fff;
`} />

### Photorealistic Rendering in Isaac Sim

Isaac Sim uses RTX technology to achieve photorealistic rendering through:

- **Path Tracing**: Simulates light transport for realistic global illumination
- **Ray Tracing**: Accurate reflection, refraction, and shadow effects
- **Physically-based Materials**: Realistic material responses to lighting
- **Volumetric Effects**: Atmospheric effects like fog and smoke

### Synthetic Data Generation

Synthetic data generation in Isaac Sim provides:

- **Ground Truth Annotations**: Pixel-perfect labels for training AI models
- **Sensor Simulation**: Accurate simulation of cameras, LiDAR, IMU, etc.
- **Domain Randomization**: Variation in lighting, textures, and objects to improve model robustness
- **Large Scale Data**: Generate thousands of images without physical limitations

### USD (Universal Scene Description)

USD is Pixar's scene description format that Isaac Sim uses:

- **Scalability**: Handle complex scenes with millions of primitives
- **Composition**: Combine multiple assets and scenes
- **Animation**: Support for complex animation and simulation data
- **Interchange**: Share assets between different 3D applications

## Step-by-Step Labs

### Lab 1: Creating a Photorealistic Environment

1. **Launch Isaac Sim** and set up a new scene:
   ```python
   # Create a new scene in Isaac Sim
   import omni
   from omni.isaac.core import World
   from omni.isaac.core.utils.stage import add_reference_to_stage
   from omni.isaac.core.utils.prims import create_prim
   from omni.isaac.core.utils.viewports import set_camera_view
   from pxr import Gf, UsdGeom, Sdf
   
   # Create a new world
   world = World(stage_units_in_meters=1.0)
   stage = omni.usd.get_context().get_stage()
   ```

2. **Create a realistic indoor environment**:
   ```python
   # Create a room environment
   def create_room_environment():
       # Create floor
       create_prim(
           "/World/floor",
           "Plane",
           position=(0, 0, 0),
           attributes={"size": 10.0}
       )
       
       # Add realistic floor material
       omni.kit.commands.execute(
           "CreateMdlMaterialPrimCommand",
           prim_path="/World/floor_material",
           mdl_file_path="OmniSurface.mdl",
           mtl_name="Omni_pbr__default",
           mtl_created_list=["/World/floor_material"]
       )
       
       # Apply material to floor
       stage = omni.usd.get_context().get_stage()
       floor_prim = stage.GetPrimAtPath("/World/floor")
       UsdGeom.MaterialBindingAPI(floor_prim).Bind(
           stage.GetPrimAtPath("/World/floor_material")
       )
       
       # Create walls
       wall_positions = [
           (0, 5, 2.5),    # North wall
           (0, -5, 2.5),   # South wall
           (5, 0, 2.5),    # East wall
           (-5, 0, 2.5)    # West wall
       ]
       
       wall_rotations = [
           (0, 0, 0),      # North wall - no rotation
           (0, 180, 0),    # South wall - 180° rotation
           (0, 90, 0),     # East wall - 90° rotation
           (0, -90, 0)     # West wall - -90° rotation
       ]
       
       for i, (pos, rot) in enumerate(zip(wall_positions, wall_rotations)):
           create_prim(
               f"/World/wall_{i}",
               "Cube",
               position=pos,
               orientation=(rot[0]*Gf.Vec3f(1, 0, 0), rot[1]*Gf.Vec3f(0, 1, 0), rot[2]*Gf.Vec3f(0, 0, 1)),
               attributes={"size": 0.2}
           )
       
       # Create ceiling
       create_prim(
           "/World/ceiling",
           "Plane",
           position=(0, 0, 5),
           attributes={"size": 10.0}
       )
   
   create_room_environment()
   ```

3. **Add realistic lighting**:
   ```python
   # Add dome light for environment lighting
   create_prim(
       "/World/DomeLight",
       "DomeLight",
       attributes={
           "color": (1.0, 1.0, 1.0),
           "intensity": 300.0,
           "texture:file": "path/to/your/hdri/environment.hdr"
       }
   )
   
   # Add rectangular light panel
   create_prim(
       "/World/RectLight",
       "RectLight",
       position=(0, 0, 4.9),  # Position just below ceiling
       attributes={
           "color": (0.9, 0.9, 1.0),  # Slightly blue-tinted white
           "intensity": 500.0,
           "width": 1.5,
           "height": 1.0
       }
   )
   ```

### Lab 2: Configuring RTX Rendering

1. **Enable RTX rendering**:
   ```python
   # Enable RTX rendering in Isaac Sim
   import carb
   
   # Set rendering kit to PathTracing
   settings = carb.settings.get_settings()
   settings.set("/rtx/renderMode", "PathTracing")
   
   # Configure Path Tracing parameters
   settings.set("/rtx/pathtracing/maxBounces", 8)
   settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 8)
   settings.set("/rtx/indirectDiffuseQuality", 2)  # High quality
   settings.set("/rtx/indirectSpecularQuality", 2)  # High quality
   ```

2. **Add physically-based materials**:
   ```python
   # Create a physically-based material
   def create_pbr_material(material_path, base_color, roughness=0.5, metallic=0.0):
       # Create material using Omniverse Material Library
       omni.kit.commands.execute(
           "CreateMdlMaterialPrimCommand",
           prim_path=material_path,
           mdl_file_path="OmniSurface.mdl",
           mtl_name="Omni_pbr__default"
       )
       
       # Set material properties
       stage = omni.usd.get_context().get_stage()
       material = stage.GetPrimAtPath(material_path)
       
       # Set base color
       material.GetAttribute("inputs:diffuse_tint").Set(base_color)
       # Set roughness
       material.GetAttribute("inputs:roughness").Set(roughness)
       # Set metallic
       material.GetAttribute("inputs:metallic").Set(metallic)
       
       return material
   
   # Create different materials for objects
   red_plastic = create_pbr_material("/World/RedPlastic", (1.0, 0.2, 0.2), roughness=0.2, metallic=0.0)
   metallic_surface = create_pbr_material("/World/MetallicSurface", (0.7, 0.7, 0.8), roughness=0.1, metallic=0.8)
   ```

### Lab 3: Adding Objects and Assets

1. **Load 3D models and assets**:
   ```python
   # Add furniture and objects to the scene
   def add_objects_to_scene():
       # Add a table
       add_reference_to_stage(
           usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/Junkyard/table.usd",
           prim_path="/World/Table"
       )
       
       # Add various objects with different materials
       objects = [
           {"name": "bottle", "usd": "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/035_power_drill.usd", "pos": (0.5, 0.5, 0.8)},
           {"name": "can", "usd": "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd", "pos": (-0.3, 0.2, 0.8)},
           {"name": "bowl", "usd": "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/025_mug.usd", "pos": (0.2, -0.4, 0.8)}
       ]
       
       for i, obj in enumerate(objects):
           add_reference_to_stage(
               usd_path=obj["usd"],
               prim_path=f"/World/Objects/{obj['name']}_{i}"
           )
           
           # Set position
           stage = omni.usd.get_context().get_stage()
           prim = stage.GetPrimAtPath(f"/World/Objects/{obj['name']}_{i}")
           xform = UsdGeom.Xformable(prim)
           xform.AddTranslateOp().Set(Gf.Vec3d(*obj["pos"]))
   
   add_objects_to_scene()
   ```

### Lab 4: Configuring Sensor Simulation

1. **Add a realistic camera sensor**:
   ```python
   # Add a USD Camera to the stage
   def add_camera(name, position, orientation, parent_path="/World"):
       camera_path = f"{parent_path}/{name}"
       
       # Create the camera prim
       stage = omni.usd.get_context().get_stage()
       camera = UsdGeom.Camera.Define(stage, camera_path)
       
       # Set camera properties
       camera.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(*position))
       camera.GetPrim().GetAttribute("xformOp:orient").Set(Gf.Quatf(*orientation))
       
       # Set camera intrinsics
       camera.GetFocalLengthAttr().Set(24.0)  # mm
       camera.GetHorizontalApertureAttr().Set(36.0)  # mm
       camera.GetVerticalApertureAttr().Set(20.25)   # mm
       
       # Set clipping range
       camera.GetClippingRangeAttr().Set((0.1, 1000.0))
       
       return camera_path
   ```

2. **Configure camera for synthetic data generation**:
   ```python
   # Add a camera for generating synthetic RGB data
   camera_path = add_camera(
       name="SyntheticCamera",
       position=(1.5, 1.5, 1.2),  # Position above the table
       orientation=(0.0, 0.0, 0.0, 1.0)  # Identity quaternion (no rotation)
   )
   
   # Configure camera for rendering
   from omni.isaac.synthetic_utils import SyntheticDataHelper
   
   # Set up synthetic data helper
   sd_helper = SyntheticDataHelper()
   sd_helper.initialize(camera_path)
   ```

## Runnable Code Example

Here's a complete example that demonstrates creating a photorealistic scene and generating synthetic data:

```python
# synthetic_data_generator.py
import omni
import carb
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom, Sdf
import numpy as np
import cv2
from omni.synthetic_utils import converters
import os

class PhotorealisticSceneBuilder:
    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.sd_helper = None
        
    def create_environment(self):
        """Create a photorealistic indoor environment"""
        print("Creating photorealistic environment...")
        
        # Create floor with realistic material
        create_prim(
            "/World/floor",
            "Plane",
            position=(0, 0, 0),
            attributes={"size": 10.0}
        )
        
        # Create walls
        wall_positions = [
            (0, 5, 2.5), (0, -5, 2.5), (5, 0, 2.5), (-5, 0, 2.5)
        ]
        wall_rotations = [
            (0, 0, 0), (0, 180, 0), (0, 90, 0), (0, -90, 0)
        ]
        
        for i, (pos, rot) in enumerate(zip(wall_positions, wall_rotations)):
            create_prim(
                f"/World/wall_{i}",
                "Cube",
                position=pos,
                attributes={"size": 0.2}
            )
        
        # Create ceiling
        create_prim(
            "/World/ceiling",
            "Plane",
            position=(0, 0, 5),
            attributes={"size": 10.0}
        )
        
        # Add lighting
        create_prim(
            "/World/DomeLight",
            "DomeLight",
            attributes={
                "color": (1.0, 1.0, 1.0),
                "intensity": 500.0
            }
        )
        
        print("Environment created successfully")
    
    def add_objects_with_randomization(self):
        """Add objects with domain randomization"""
        print("Adding objects with randomization...")
        
        # Object models to choose from
        object_paths = [
            "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
            "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
            "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
            "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd"
        ]
        
        # Add 5 random objects
        for i in range(5):
            # Random position within bounds
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            z = 0.1  # Just above floor
            
            # Add object to scene
            add_reference_to_stage(
                usd_path=np.random.choice(object_paths),
                prim_path=f"/World/Object_{i}"
            )
            
            # Set random position
            prim = self.stage.GetPrimAtPath(f"/World/Object_{i}")
            xform = UsdGeom.Xformable(prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            
            # Random rotation
            xform.AddRotateXYZOp().Set(Gf.Vec3f(
                np.random.uniform(0, 360),
                np.random.uniform(0, 360),
                np.random.uniform(0, 360)
            ))
        
        print("Objects added successfully")
    
    def setup_camera_for_synthetic_data(self):
        """Set up camera for synthetic data generation"""
        print("Setting up camera for synthetic data...")
        
        # Create camera
        camera_path = "/World/RGB_Camera"
        camera = UsdGeom.Camera.Define(self.stage, camera_path)
        
        # Position camera
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(2, 2, 1.5))
        
        # Point camera toward center
        set_camera_view(eye=(2, 2, 1.5), target=(0, 0, 1))
        
        # Set camera properties
        camera.GetFocalLengthAttr().Set(24.0)
        camera.GetHorizontalApertureAttr().Set(36.0)
        camera.GetVerticalApertureAttr().Set(20.25)
        camera.GetClippingRangeAttr().Set((0.1, 1000.0))
        
        print("Camera setup complete")
        
        return camera_path
    
    def enable_rtx_rendering(self):
        """Configure RTX rendering settings"""
        print("Enabling RTX rendering...")
        
        settings = carb.settings.get_settings()
        
        # Set to Path Tracing mode for highest quality
        settings.set("/rtx/renderMode", "PathTracing")
        
        # Set high quality parameters
        settings.set("/rtx/pathtracing/maxBounces", 8)
        settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 8)
        settings.set("/rtx/indirectDiffuseQuality", 2)
        settings.set("/rtx/indirectSpecularQuality", 2)
        settings.set("/rtx/domeLight/enable", True)
        
        print("RTX rendering enabled")
    
    def capture_synthetic_data(self, output_dir="synthetic_data", num_frames=10):
        """Capture synthetic RGB and depth data"""
        print(f"Capturing {num_frames} synthetic data frames...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/seg", exist_ok=True)  # segmentation
        
        # Get the USD camera
        camera_path = "/World/RGB_Camera"
        
        # Wait for rendering to initialize
        import time
        time.sleep(2)
        
        for frame_idx in range(num_frames):
            print(f"Capturing frame {frame_idx+1}/{num_frames}")
            
            # Randomize lighting and object positions for domain randomization
            self.randomize_scene()
            
            # Wait for frame to render
            from omni.isaac.core.utils.viewports import get_viewport_from_window_name
            viewport_api = get_viewport_from_window_name("Viewport")
            
            # Trigger capture (this is a simplified approach)
            # In a real implementation, you would use the Synthetic Data API
            self.capture_frame_data(f"{output_dir}/rgb/frame_{frame_idx:04d}.png", 
                                   f"{output_dir}/depth/frame_{frame_idx:04d}.png",
                                   f"{output_dir}/seg/frame_{frame_idx:04d}.png")
            
            time.sleep(0.5)  # Wait between captures
        
        print(f"Synthetic data captured to {output_dir}")
    
    def randomize_scene(self):
        """Apply domain randomization to improve dataset robustness"""
        # Randomize dome light intensity
        dome_light = self.stage.GetPrimAtPath("/World/DomeLight")
        new_intensity = np.random.uniform(300, 800)
        dome_light.GetAttribute("inputs:intensity").Set(new_intensity)
        
        # Randomize dome light color temperature (simplified)
        new_color = (
            np.random.uniform(0.8, 1.0),
            np.random.uniform(0.8, 1.0),
            np.random.uniform(0.9, 1.0)
        )
        dome_light.GetAttribute("inputs:color").Set(new_color)
        
        # Randomize object materials
        for i in range(5):
            obj_prim = self.stage.GetPrimAtPath(f"/World/Object_{i}")
            if obj_prim:
                # In a real implementation, you would apply random materials
                # This is a placeholder for material randomization
                pass
    
    def capture_frame_data(self, rgb_path, depth_path, seg_path):
        """Capture RGB, depth, and segmentation data"""
        # In a real Isaac Sim implementation, this would use the 
        # Synthetic Data API to capture data directly
        # For this example, we'll create placeholder images
        
        # Create placeholder RGB image
        rgb_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
        # Create placeholder depth image
        depth_img = np.random.uniform(0.1, 10.0, (720, 1280)).astype(np.float32)
        cv2.imwrite(depth_path, depth_img)
        
        # Create placeholder segmentation image
        seg_img = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)
        cv2.imwrite(seg_path, seg_img)
        
        print(f"Captured data to {rgb_path}, {depth_path}, {seg_path}")

def main():
    """Main function to create photorealistic scene and generate data"""
    # Initialize Isaac Sim context
    print("Initializing Isaac Sim environment...")
    
    # Create scene builder
    builder = PhotorealisticSceneBuilder()
    
    # Create photorealistic environment
    builder.create_environment()
    
    # Add objects with randomization
    builder.add_objects_with_randomization()
    
    # Enable RTX rendering
    builder.enable_rtx_rendering()
    
    # Set up camera for synthetic data
    camera_path = builder.setup_camera_for_synthetic_data()
    
    # Generate synthetic data
    builder.capture_synthetic_data(num_frames=5)
    
    print("Synthetic data generation complete!")

# Isaac Sim Extension
class SyntheticDataExtension:
    def __init__(self):
        self.scene_builder = PhotorealisticSceneBuilder()
    
    def build_photorealistic_scene(self):
        self.scene_builder.create_environment()
        self.scene_builder.add_objects_with_randomization()
        self.scene_builder.enable_rtx_rendering()
        self.scene_builder.setup_camera_for_synthetic_data()
    
    def generate_dataset(self, config):
        """Generate a complete synthetic dataset based on config"""
        # Apply domain randomization settings from config
        num_frames = config.get("num_frames", 100)
        output_dir = config.get("output_dir", "synthetic_dataset")
        scene_complexity = config.get("scene_complexity", "medium")
        
        # Create appropriate environment based on complexity
        if scene_complexity == "simple":
            # Simple scene with few objects
            pass
        elif scene_complexity == "medium":
            # Medium complexity scene
            pass
        elif scene_complexity == "complex":
            # Complex scene with many objects and lighting variations
            pass
        
        # Generate the dataset
        self.scene_builder.capture_synthetic_data(
            output_dir=output_dir,
            num_frames=num_frames
        )

if __name__ == "__main__":
    main()
```

### Isaac Sim Python Script for Synthetic Data Generation

```python
# advanced_synthetic_data.py
import omni
import carb
from pxr import Gf, UsdGeom, Sdf
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.synthetic_utils import shaders
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from omni.kit.viewport.utility import get_active_viewport
import numpy as np
import PIL.Image
import os

class AdvancedSyntheticDataGenerator:
    def __init__(self):
        self.sd_helper = SyntheticDataHelper()
        self.stage = omni.usd.get_context().get_stage()
        self.viewport = get_active_viewport()
        
    def setup_sensor_suite(self, camera_path):
        """Set up multiple sensors for comprehensive data capture"""
        # Initialize synthetic data helper with the camera
        self.sd_helper.initialize(camera_path)
        
        # Enable different types of synthetic data
        try:
            # RGB data
            self.sd_helper.get_rgb()
            
            # Depth data (world units)
            self.sd_helper.get_depth()
            
            # Semantic segmentation
            self.sd_helper.get_semantic_segmentation()
            
            # Instance segmentation
            self.sd_helper.get_instance_segmentation()
            
            # Normals
            self.sd_helper.get_normals()
            
            # Motion vectors
            self.sd_helper.get_motion_vectors()
            
            print("All sensor types enabled")
        except Exception as e:
            print(f"Error setting up sensors: {e}")
    
    def generate_domain_randomization_config(self):
        """Generate configuration for domain randomization"""
        config = {
            # Lighting randomization
            "lighting": {
                "intensity_range": (300, 800),
                "color_temperature_range": (5000, 8000),  # Kelvin
                "shadow_softness_range": (0.1, 0.8)
            },
            
            # Material randomization
            "materials": {
                "albedo_range": (0.1, 1.0),
                "roughness_range": (0.0, 1.0),
                "metallic_range": (0.0, 1.0),
                "normal_map_strength_range": (0.0, 1.0)
            },
            
            # Object placement
            "objects": {
                "position_jitter": 0.1,  # meters
                "rotation_jitter": 5.0,  # degrees
                "scale_range": (0.8, 1.2)
            },
            
            # Camera randomization
            "camera": {
                "position_jitter": 0.05,
                "orientation_jitter": 2.0,
                "focal_length_range": (18, 55)  # mm equivalent
            }
        }
        
        return config
    
    def apply_domain_randomization(self, config):
        """Apply domain randomization to the scene"""
        # Randomize lighting
        dome_light = self.stage.GetPrimAtPath("/World/DomeLight")
        if dome_light:
            # Random intensity
            intensity = np.random.uniform(
                config["lighting"]["intensity_range"][0], 
                config["lighting"]["intensity_range"][1]
            )
            dome_light.GetAttribute("inputs:intensity").Set(intensity)
            
            # Random color temperature (simplified)
            color_temp = np.random.uniform(
                config["lighting"]["color_temperature_range"][0],
                config["lighting"]["color_temperature_range"][1]
            )
            # Convert to RGB approximation
            rgb = self.color_temperature_to_rgb(color_temp)
            dome_light.GetAttribute("inputs:color").Set(rgb)
        
        # Randomize materials (simplified example)
        for i in range(5):
            obj_prim = self.stage.GetPrimAtPath(f"/World/Object_{i}")
            if obj_prim:
                # In a real implementation, you would apply random materials
                # This is a placeholder for material randomization
                pass
    
    def color_temperature_to_rgb(self, color_temp):
        """Convert color temperature in Kelvin to RGB (simplified)"""
        # This is a simplified approximation
        temp = color_temp / 100
        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)
        
        blue = temp - 10
        if temp >= 66:
            blue = 255
        elif temp <= 19:
            blue = 0
        else:
            blue = temp - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
        
        # Clamp values to [0, 255] and normalize to [0, 1]
        red = np.clip(red, 0, 255) / 255.0
        green = np.clip(green, 0, 255) / 255.0
        blue = np.clip(blue, 0, 255) / 255.0
        
        return (red, green, blue)
    
    def capture_synthetic_dataset(self, output_dir, num_samples, config=None):
        """Capture a comprehensive synthetic dataset"""
        if config is None:
            config = self.generate_domain_randomization_config()
        
        # Create output directories
        os.makedirs(f"{output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/seg", exist_ok=True)
        os.makedirs(f"{output_dir}/instances", exist_ok=True)
        os.makedirs(f"{output_dir}/normals", exist_ok=True)
        
        camera_path = "/World/RGB_Camera"
        self.setup_sensor_suite(camera_path)
        
        for i in range(num_samples):
            print(f"Capturing sample {i+1}/{num_samples}")
            
            # Apply domain randomization
            self.apply_domain_randomization(config)
            
            # Wait for the scene to update
            import time
            time.sleep(0.1)
            
            # Capture all data types
            self.capture_single_sample(f"{output_dir}", i)
        
        print(f"Dataset captured to {output_dir}")
    
    def capture_single_sample(self, output_dir, sample_idx):
        """Capture a single sample with all data types"""
        try:
            # Get data from synthetic data helper
            rgb_data = self.sd_helper.get_rgb()
            depth_data = self.sd_helper.get_depth()
            seg_data = self.sd_helper.get_semantic_segmentation()
            instance_data = self.sd_helper.get_instance_segmentation()
            normal_data = self.sd_helper.get_normals()
            
            # Save RGB image
            if rgb_data is not None:
                rgb_img = PIL.Image.fromarray((rgb_data * 255).astype(np.uint8))
                rgb_img.save(f"{output_dir}/rgb/{sample_idx:06d}.png")
            
            # Save depth image
            if depth_data is not None:
                depth_img = PIL.Image.fromarray((depth_data * 1000).astype(np.uint16))  # Scale for 16-bit
                depth_img.save(f"{output_dir}/depth/{sample_idx:06d}.png")
            
            # Save segmentation
            if seg_data is not None:
                seg_img = PIL.Image.fromarray((seg_data).astype(np.uint8))
                seg_img.save(f"{output_dir}/seg/{sample_idx:06d}.png")
            
            # Save instance segmentation
            if instance_data is not None:
                instance_img = PIL.Image.fromarray((instance_data).astype(np.uint8))
                instance_img.save(f"{output_dir}/instances/{sample_idx:06d}.png")
            
            # Save normals (as RGB or separate channels)
            if normal_data is not None:
                # Normalize normals to [0, 255]
                normal_scaled = ((normal_data + 1) * 0.5 * 255).astype(np.uint8)
                normal_img = PIL.Image.fromarray(normal_scaled)
                normal_img.save(f"{output_dir}/normals/{sample_idx:06d}.png")
            
        except Exception as e:
            carb.log_error(f"Error capturing sample {sample_idx}: {str(e)}")

def create_photorealistic_scene_and_generate_data():
    """Complete workflow: create scene and generate synthetic data"""
    # Initialize Isaac Sim components
    generator = AdvancedSyntheticDataGenerator()
    
    # Create or use an existing photorealistic scene
    # (In practice, you'd build your scene with realistic assets and materials)
    
    # Define dataset configuration
    dataset_config = {
        "num_samples": 100,
        "output_dir": "./synthetic_dataset",
        "domain_randomization": {
            "lighting": {"intensity_range": (300, 800)},
            "objects": {"position_jitter": 0.05}
        }
    }
    
    # Generate the dataset
    generator.capture_synthetic_dataset(
        output_dir=dataset_config["output_dir"],
        num_samples=dataset_config["num_samples"],
        config=dataset_config["domain_randomization"]
    )
    
    print("Synthetic dataset generation complete!")

# Run the workflow
if __name__ == "__main__":
    create_photorealistic_scene_and_generate_data()
```

## Mini-project

Create a complete synthetic data generation pipeline that:

1. Builds a photorealistic kitchen environment in Isaac Sim
2. Adds domain randomization for lighting, materials, and object placement
3. Configures multiple sensors (RGB, depth, semantic segmentation)
4. Generates at least 500 synthetic images with ground truth annotations
5. Implements a validation system to check data quality
6. Creates training scripts that use the synthetic data to train a simple perception model
7. Compares performance of models trained on synthetic vs real data

Your project should include:
- Complete Isaac Sim scene with photorealistic rendering
- Domain randomization system
- Data capture pipeline with multiple sensors
- Quality validation tools
- Training pipeline that demonstrates the value of synthetic data

## Summary

This chapter covered photorealistic simulation and synthetic data generation with Isaac Sim:

- **Photorealistic Rendering**: Using RTX technology and PhysX for realistic simulations
- **USD Workflows**: Understanding Universal Scene Description for scalable scene management
- **Synthetic Data Generation**: Creating ground truth datasets for AI training
- **Domain Randomization**: Techniques to improve model robustness through variation
- **Sensor Simulation**: Configuring realistic camera, LiDAR, and other sensor models
- **Data Pipeline**: Complete workflow from scene creation to dataset generation

Isaac Sim provides powerful tools for creating realistic simulation environments and generating high-quality synthetic data essential for training robust AI models in robotics applications.