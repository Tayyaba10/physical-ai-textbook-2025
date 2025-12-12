---
title: Ch9 - Unity HDRP Visualization & HRI
module: 2
chapter: 9
sidebar_label: Ch9: Unity HDRP Visualization & HRI
description: Creating high-fidelity visualizations with Unity HDRP and implementing Human-Robot Interaction
tags: [unity, hdrp, visualization, hri, human-robot-interaction, graphics, simulation]
difficulty: advanced
estimated_duration: 120
---

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Unity HDRP Visualization & HRI

## Learning Outcomes
- Understand Unity HDRP (High Definition Render Pipeline) for photorealistic visualization
- Create high-fidelity robotic simulation environments in Unity
- Implement Human-Robot Interaction (HRI) interfaces
- Integrate Unity with ROS 2 for bidirectional communication
- Design intuitive user interfaces for robot control and monitoring
- Optimize Unity scenes for real-time performance with complex robotics scenarios
- Implement VR/AR interfaces for enhanced Human-Robot Interaction

## Theory

### Unity HDRP Overview

The High Definition Render Pipeline (HDRP) is Unity's modern rendering pipeline designed for high-fidelity graphics. It offers advanced features like:

<MermaidDiagram chart={`
graph TD;
    A[Unity HDRP] --> B[Photorealistic Rendering];
    A --> C[Global Illumination];
    A --> D[Advanced Lighting];
    A --> E[Post-Processing];
    
    B --> F[Realistic Materials];
    B --> G[Accurate Reflections];
    C --> H[Volumetric Lighting];
    C --> I[Light Transport];
    D --> J[Physical Cameras];
    D --> K[Real-time Shadows];
    E --> L[Color Grading];
    E --> M[Anti-Aliasing];
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style B fill:#2196F3,stroke:#0D47A1,color:#fff;
`} />

- **Physically-Based Shading (PBS)**: Materials that respond realistically to lighting conditions
- **Global Illumination**: Realistic light scattering and bounce lighting
- **Volumetric Lighting**: Realistic light rays and atmospheric effects
- **Real-time Ray Tracing**: Accurate reflections and shadows
- **Advanced Post-Processing Effects**: Color grading, depth of field, motion blur

### Human-Robot Interaction (HRI) Principles

HRI in Unity simulation environments involves creating intuitive interfaces that allow humans to interact with robots in realistic scenarios:

- **Visual Feedback**: Clear representation of robot state, intentions, and status
- **Intuitive Controls**: Natural and responsive interfaces for commanding robots
- **Situation Awareness**: Providing users with necessary environmental and robot information
- **Safety Considerations**: Interfaces that promote safe human-robot collaboration

### Unity-ROS Integration

Unity can communicate with ROS 2 through various approaches:

1. **ROS# Library**: Direct integration allowing Unity to act as a ROS node
2. **WebSocket Bridge**: Communication via web sockets
3. **TCP/UDP Bridge**: Direct socket communication
4. **ROS Bridge Package**: Using the official ROS Bridge package

## Step-by-Step Labs

### Lab 1: Setting up Unity HDRP Project

1. **Create a new Unity HDRP project**:
   - Open Unity Hub
   - Create new project
   - Select "3D (High Definition Render Pipeline)" template
   - Or create a 3D project and convert to HDRP later

2. **Configure the HDRP asset**:
   - Create an HDRP asset in your Assets folder
   - Configure settings like lighting, post-processing, and performance parameters

3. **Basic HDRP setup script** (`HDRPSetup.cs`):
   ```csharp
   using UnityEngine;
   using UnityEngine.Rendering;
   using UnityEngine.Rendering.HighDefinition;

   public class HDRPSetup : MonoBehaviour
   {
       [Header("Lighting Settings")]
       public float globalLightIntensity = 3.14f;
       public Color globalLightColor = Color.white;
       
       [Header("Post-Processing Settings")]
       public bool useBloom = true;
       public float bloomIntensity = 0.5f;
       public bool useDepthOfField = true;
       
       private HDAdditionalLightData mainLightData;
       private HDAdditionalCameraData cameraData;
       
       void Start()
       {
           SetupLighting();
           SetupCamera();
           SetupPostProcessing();
       }
       
       void SetupLighting()
       {
           // Find the main directional light
           Light mainLight = RenderSettings.sun;
           if (mainLight != null)
           {
               mainLightData = mainLight.GetComponent<HDAdditionalLightData>();
               if (mainLightData != null)
               {
                   mainLight.intensity = globalLightIntensity;
                   mainLight.color = globalLightColor;
                   mainLightData.SetHDAdditionalLightData(HDLightType.Directional, globalLightIntensity, globalLightColor);
               }
           }
       }
       
       void SetupCamera()
       {
           Camera mainCamera = Camera.main;
           if (mainCamera != null)
           {
               cameraData = mainCamera.GetComponent<HDAdditionalCameraData>();
               if (cameraData != null)
               {
                   // Configure camera settings for simulation
                   cameraData.customRenderingSettings = true;
                   cameraData.renderingPath = RenderingPath.HighDefinition;
               }
           }
       }
       
       void SetupPostProcessing()
       {
           // Apply post-processing settings
           if (useBloom)
           {
               // Bloom settings would be configured here
               // This is a simplified example - actual implementation would use Volume components
           }
       }
   }
   ```

### Lab 2: Creating a Robot Visualization

1. **Create a basic robot model** in Unity:
   - Use primitives (cylinders, cubes, spheres) to create a simple robot
   - Or import a 3D model of a robot

2. **Create robot controller script** (`RobotController.cs`):
   ```csharp
   using UnityEngine;

   [System.Serializable]
   public class JointConfig
   {
       public string jointName;
       public float minAngle = -90f;
       public float maxAngle = 90f;
       public float currentAngle = 0f;
   }

   public class RobotController : MonoBehaviour
   {
       [Header("Robot Configuration")]
       public JointConfig[] joints;
       
       [Header("Visual Components")]
       public Material defaultMaterial;
       public Material selectedMaterial;
       public Renderer[] robotParts;
       
       [Header("Simulation Parameters")]
       public float movementSpeed = 1.0f;
       public float rotationSpeed = 50.0f;
       
       private bool isConnected = false;
       
       void Start()
       {
           InitializeRobot();
       }
       
       void Update()
       {
           // Update joint positions based on config
           UpdateJointPositions();
           
           // Handle robot movement if connected to simulation
           if (isConnected)
           {
               ProcessSimulationInputs();
           }
       }
       
       void InitializeRobot()
       {
           // Initialize joints with default values
           for (int i = 0; i < joints.Length; i++)
           {
               joints[i].currentAngle = 0f;
           }
       }
       
       void UpdateJointPositions()
       {
           // This would typically be called with data from ROS
           // For now, we'll just apply the stored angles
           for (int i = 0; i < joints.Length; i++)
           {
               Transform jointTransform = FindChildByName(joints[i].jointName);
               if (jointTransform != null)
               {
                   // Apply rotation based on joint configuration
                   Vector3 rotation = jointTransform.localEulerAngles;
                   // This is a simplification - real implementation would need to know which axis to rotate
                   jointTransform.localEulerAngles = new Vector3(rotation.x, rotation.y, joints[i].currentAngle);
               }
           }
       }
       
       Transform FindChildByName(string name)
       {
           Transform[] children = GetComponentsInChildren<Transform>();
           foreach (Transform child in children)
           {
               if (child.name == name)
                   return child;
           }
           return null;
       }
       
       public void SetJointAngle(string jointName, float angle)
       {
           for (int i = 0; i < joints.Length; i++)
           {
               if (joints[i].jointName == jointName)
               {
                   joints[i].currentAngle = Mathf.Clamp(angle, joints[i].minAngle, joints[i].maxAngle);
                   break;
               }
           }
       }
       
       public void ConnectToSimulation()
       {
           isConnected = true;
       }
       
       public void DisconnectFromSimulation()
       {
           isConnected = false;
       }
       
       void ProcessSimulationInputs()
       {
           // This would handle inputs from ROS simulation
           // For now, just a placeholder
       }
       
       // Visualization of robot state
       public void HighlightRobot()
       {
           if (robotParts != null)
           {
               foreach (Renderer part in robotParts)
               {
                   if (part != null)
                   {
                       part.material = selectedMaterial;
                   }
               }
           }
       }
       
       public void ResetRobotVisuals()
       {
           if (robotParts != null)
           {
               foreach (Renderer part in robotParts)
               {
                   if (part != null)
                   {
                       part.material = defaultMaterial;
                   }
               }
           }
       }
   }
   ```

### Lab 3: Implementing Human-Robot Interaction Interface

1. **Create HRI UI controller** (`HRIController.cs`):
   ```csharp
   using UnityEngine;
   using UnityEngine.UI;
   using TMPro;

   public class HRIController : MonoBehaviour
   {
       [Header("UI Elements")]
       public Button connectButton;
       public Button disconnectButton;
       public Button moveForwardButton;
       public Button moveBackwardButton;
       public Button turnLeftButton;
       public Button turnRightButton;
       public Slider speedSlider;
       public TextMeshProUGUI statusText;
       public TextMeshProUGUI jointStatusText;
       
       [Header("Robot Reference")]
       public RobotController robot;
       
       [Header("Command Parameters")]
       public float linearSpeed = 1.0f;
       public float angularSpeed = 1.0f;
       
       void Start()
       {
           SetupUI();
       }
       
       void SetupUI()
       {
           if (connectButton != null)
               connectButton.onClick.AddListener(ConnectToRobot);
               
           if (disconnectButton != null)
               disconnectButton.onClick.AddListener(DisconnectFromRobot);
               
           if (moveForwardButton != null)
               moveForwardButton.onClick.AddListener(MoveForward);
               
           if (moveBackwardButton != null)
               moveBackwardButton.onClick.AddListener(MoveBackward);
               
           if (turnLeftButton != null)
               turnLeftButton.onClick.AddListener(TurnLeft);
               
           if (turnRightButton != null)
               turnRightButton.onClick.AddListener(TurnRight);
               
           if (speedSlider != null)
               speedSlider.onValueChanged.AddListener(UpdateSpeed);
       }
       
       void ConnectToRobot()
       {
           if (robot != null)
           {
               robot.ConnectToSimulation();
               UpdateStatus("Connected to robot");
           }
       }
       
       void DisconnectFromRobot()
       {
           if (robot != null)
           {
               robot.DisconnectFromSimulation();
               UpdateStatus("Disconnected from robot");
           }
       }
       
       void MoveForward()
       {
           if (robot != null && robot.GetComponent<Rigidbody>() != null)
           {
               Rigidbody rb = robot.GetComponent<Rigidbody>();
               rb.AddForce(robot.transform.forward * linearSpeed, ForceMode.VelocityChange);
               UpdateStatus("Moving forward");
           }
       }
       
       void MoveBackward()
       {
           if (robot != null && robot.GetComponent<Rigidbody>() != null)
           {
               Rigidbody rb = robot.GetComponent<Rigidbody>();
               rb.AddForce(-robot.transform.forward * linearSpeed, ForceMode.VelocityChange);
               UpdateStatus("Moving backward");
           }
       }
       
       void TurnLeft()
       {
           if (robot != null)
           {
               robot.transform.Rotate(Vector3.up, -angularSpeed * Time.deltaTime, Space.World);
               UpdateStatus("Turning left");
           }
       }
       
       void TurnRight()
       {
           if (robot != null)
           {
               robot.transform.Rotate(Vector3.up, angularSpeed * Time.deltaTime, Space.World);
               UpdateStatus("Turning right");
           }
       }
       
       void UpdateSpeed(float newSpeed)
       {
           linearSpeed = newSpeed;
           angularSpeed = newSpeed;
           UpdateStatus($"Speed updated: {newSpeed:F2}");
       }
       
       void UpdateStatus(string status)
       {
           if (statusText != null)
           {
               statusText.text = $"Status: {status}";
           }
       }
       
       void UpdateJointStatus()
       {
           if (jointStatusText != null && robot != null)
           {
               string jointInfo = "Joint Status:\n";
               foreach (var joint in robot.joints)
               {
                   jointInfo += $"{joint.jointName}: {joint.currentAngle:F2}°\n";
               }
               jointStatusText.text = jointInfo;
           }
       }
       
       void Update()
       {
           // Update joint status continuously
           UpdateJointStatus();
       }
   }
   ```

### Lab 4: ROS Integration (Conceptual)

While Unity doesn't have native ROS support, here's a conceptual implementation showing how you might integrate with ROS:

1. **ROS Communication Manager** (`ROSCommunicationManager.cs`):
   ```csharp
   using System.Collections;
   using System.Collections.Generic;
   using UnityEngine;
   using System.Net.WebSockets;
   using System.Threading;
   using System.Threading.Tasks;
   using System.Text;
   
   // This is a simplified example - a real implementation would require more sophisticated networking
   public class ROSCommunicationManager : MonoBehaviour
   {
       [Header("ROS Connection Settings")]
       public string rosBridgeUrl = "ws://localhost:9090";
       public float updateRate = 10.0f;  // Hz
       
       private ClientWebSocket webSocket;
       private bool isConnected = false;
       private CancellationTokenSource cancellationTokenSource;
       
       [Header("Robot Reference")]
       public RobotController robotController;
       
       void Start()
       {
           ConnectToROSBridge();
       }
       
       async void ConnectToROSBridge()
       {
           try
           {
               webSocket = new ClientWebSocket();
               cancellationTokenSource = new CancellationTokenSource();
               
               await webSocket.ConnectAsync(new System.Uri(rosBridgeUrl), cancellationTokenSource.Token);
               isConnected = true;
               
               // Start receiving messages
               StartCoroutine(ReceiveMessages());
               
               Debug.Log("Connected to ROS Bridge");
           }
           catch (System.Exception e)
           {
               Debug.LogError($"Failed to connect to ROS Bridge: {e.Message}");
           }
       }
       
       IEnumerator ReceiveMessages()
       {
           // This would run continuously to receive messages from ROS
           while (isConnected && webSocket.State == WebSocketState.Open)
           {
               // In a real implementation, you would receive ROS messages here
               // and update the Unity scene accordingly
               
               // For now, just simulate receiving joint states
               if (robotController != null)
               {
                   // Simulate receiving joint state data
                   SimulateJointStateUpdate();
               }
               
               yield return new WaitForSeconds(1.0f / updateRate);
           }
       }
       
       void SimulateJointStateUpdate()
       {
           // Simulate updating robot joints based on ROS messages
           if (robotController != null)
           {
               // In a real implementation, this would process actual ROS JointState messages
               // For simulation, we'll just update with realistic values
               foreach (var joint in robotController.joints)
               {
                   // Add small random changes to simulate real robot movement
                   joint.currentAngle += Random.Range(-2f, 2f);
                   joint.currentAngle = Mathf.Clamp(joint.currentAngle, joint.minAngle, joint.maxAngle);
               }
           }
       }
       
       async void SendCommand(string topic, string message)
       {
           if (isConnected && webSocket.State == WebSocketState.Open)
           {
               byte[] messageBytes = Encoding.UTF8.GetBytes(message);
               await webSocket.SendAsync(new ArraySegment<byte>(messageBytes), WebSocketMessageType.Text, true, cancellationTokenSource.Token);
           }
       }
       
       void OnApplicationQuit()
       {
           if (cancellationTokenSource != null)
           {
               cancellationTokenSource.Cancel();
           }
           
           if (webSocket != null)
           {
               webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Application closing", CancellationToken.None);
           }
       }
   }
   ```

## Runnable Code Example

Here's a complete example of a Unity-based visualization system for a robotic arm:

```csharp
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;
using System.Collections.Generic;

// Main class to manage the robotic arm visualization
public class RoboticArmVisualizer : MonoBehaviour
{
    [Header("Arm Configuration")]
    public Transform baseJoint;      // Base of the arm
    public Transform[] joints;       // All joints of the arm
    public Transform[] links;        // All links of the arm
    public Transform endEffector;    // End effector of the arm
    
    [Header("Joint Limits")]
    public float[] minJointAngles = new float[6];
    public float[] maxJointAngles = new float[6];
    public float[] currentJointAngles = new float[6];
    
    [Header("Visualization Parameters")]
    public Material defaultMaterial;
    public Material selectedMaterial;
    public float animationSpeed = 5f;
    public bool useIK = false;
    
    [Header("Target Tracking")]
    public Transform targetObject;
    public bool trackTarget = false;
    
    private List<Renderer> armRenderers = new List<Renderer>();
    private bool isAnimating = false;
    
    void Start()
    {
        InitializeArm();
        SetupMaterials();
    }
    
    void Update()
    {
        if (trackTarget && targetObject != null)
        {
            UpdateArmToTarget();
        }
        else if (!isAnimating)
        {
            UpdateArmPosition();
        }
    }
    
    void InitializeArm()
    {
        // Initialize joint angles
        for (int i = 0; i < currentJointAngles.Length; i++)
        {
            if (i < joints.Length)
            {
                currentJointAngles[i] = joints[i].localEulerAngles.y;  // Assuming rotation around Y axis
            }
        }
        
        // Collect all renderers for material changes
        armRenderers.AddRange(GetComponentsInChildren<Renderer>());
    }
    
    void SetupMaterials()
    {
        if (defaultMaterial != null)
        {
            foreach (Renderer renderer in armRenderers)
            {
                if (renderer != null)
                {
                    renderer.material = defaultMaterial;
                }
            }
        }
    }
    
    public void SetJointAngles(float[] angles)
    {
        if (angles.Length == currentJointAngles.Length)
        {
            for (int i = 0; i < angles.Length; i++)
            {
                currentJointAngles[i] = Mathf.Clamp(angles[i], minJointAngles[i], maxJointAngles[i]);
            }
        }
        else
        {
            Debug.LogWarning("Joint angle array size mismatch");
        }
    }
    
    public void SetJointAngle(int jointIndex, float angle)
    {
        if (jointIndex >= 0 && jointIndex < currentJointAngles.Length)
        {
            currentJointAngles[jointIndex] = Mathf.Clamp(angle, minJointAngles[jointIndex], maxJointAngles[jointIndex]);
        }
    }
    
    void UpdateArmPosition()
    {
        // Update each joint based on its target angle
        for (int i = 0; i < joints.Length && i < currentJointAngles.Length; i++)
        {
            if (joints[i] != null)
            {
                // Animate joint rotation towards target angle
                Vector3 currentRotation = joints[i].localEulerAngles;
                
                // Calculate new angle with smooth interpolation
                float newAngle = Mathf.LerpAngle(
                    currentRotation.y, 
                    currentJointAngles[i], 
                    Time.deltaTime * animationSpeed
                );
                
                joints[i].localEulerAngles = new Vector3(currentRotation.x, newAngle, currentRotation.z);
            }
        }
    }
    
    void UpdateArmToTarget()
    {
        if (targetObject == null || endEffector == null) return;
        
        if (useIK)
        {
            // Simple inverse kinematics for reaching target
            // This is a simplified implementation - real IK would be more complex
            Vector3 directionToTarget = (targetObject.position - baseJoint.position).normalized;
            
            // Set the first joint to point toward the target
            if (joints.Length > 0 && joints[0] != null)
            {
                joints[0].transform.LookAt(targetObject);
                joints[0].transform.Rotate(-90, 0, 0); // Adjust for joint orientation
            }
            
            // Additional IK calculations would go here for more complex arms
        }
        else
        {
            // Move arm closer to target position (simplified)
            Vector3 directionToTarget = (targetObject.position - endEffector.position).normalized;
            float distance = Vector3.Distance(endEffector.position, targetObject.position);
            
            if (distance > 0.1f) // Threshold for reaching the target
            {
                // Adjust joint angles to move closer to target
                // This is a simplified approach - real implementation would use proper IK algorithms
                for (int i = 0; i < joints.Length; i++)
                {
                    if (joints[i] != null)
                    {
                        // Rotate joint slightly toward target position
                        float adjustment = Mathf.Sign(Vector3.Dot(directionToTarget, joints[i].transform.forward)) * 0.5f;
                        SetJointAngle(i, currentJointAngles[i] + adjustment);
                    }
                }
            }
        }
    }
    
    public void HighlightArm()
    {
        if (selectedMaterial != null)
        {
            foreach (Renderer renderer in armRenderers)
            {
                if (renderer != null)
                {
                    renderer.material = selectedMaterial;
                }
            }
        }
    }
    
    public void ResetArmVisuals()
    {
        if (defaultMaterial != null)
        {
            foreach (Renderer renderer in armRenderers)
            {
                if (renderer != null)
                {
                    renderer.material = defaultMaterial;
                }
            }
        }
    }
    
    // Animation coroutine for smooth movement
    public IEnumerator AnimateToJointConfiguration(float[] targetAngles, float duration)
    {
        isAnimating = true;
        
        float[] startAngles = new float[currentJointAngles.Length];
        for (int i = 0; i < currentJointAngles.Length; i++)
        {
            startAngles[i] = currentJointAngles[i];
        }
        
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float progress = Mathf.Clamp01(elapsed / duration);
            
            for (int i = 0; i < currentJointAngles.Length; i++)
            {
                currentJointAngles[i] = Mathf.Lerp(startAngles[i], targetAngles[i], progress);
            }
            
            yield return null;
        }
        
        // Ensure final position is reached
        for (int i = 0; i < currentJointAngles.Length; i++)
        {
            currentJointAngles[i] = targetAngles[i];
        }
        
        isAnimating = false;
    }
    
    // Gizmos for debugging joint positions
    void OnDrawGizmos()
    {
        if (joints != null)
        {
            for (int i = 0; i < joints.Length; i++)
            {
                if (joints[i] != null)
                {
                    Gizmos.color = Color.yellow;
                    Gizmos.DrawSphere(joints[i].position, 0.05f);
                    
                    // Draw line to next joint
                    if (i < joints.Length - 1 && joints[i+1] != null)
                    {
                        Gizmos.color = Color.cyan;
                        Gizmos.DrawLine(joints[i].position, joints[i+1].position);
                    }
                }
            }
            
            // Draw end effector
            if (endEffector != null)
            {
                Gizmos.color = Color.red;
                Gizmos.DrawSphere(endEffector.position, 0.05f);
            }
            
            // Draw target if tracking
            if (trackTarget && targetObject != null)
            {
                Gizmos.color = Color.green;
                Gizmos.DrawSphere(targetObject.position, 0.1f);
                Gizmos.DrawLine(endEffector != null ? endEffector.position : Vector3.zero, targetObject.position);
            }
        }
    }
}
```

### Unity Scene Setup for Robotics Visualization

The scene setup would include:

1. **Environment Setup**: Create a realistic environment with proper lighting
2. **Robot Model**: Import or create robot models with proper physical properties
3. **Camera System**: Set up cameras for different viewing perspectives
4. **Lighting**: Configure HDRP lighting for realistic appearance
5. **Post-Processing**: Apply post-processing effects for enhanced visual quality

### HRI Dashboard Example (`HRIDashboard.cs`)

```csharp
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

public class HRIDashboard : MonoBehaviour
{
    [Header("Dashboard Elements")]
    public TextMeshProUGUI robotNameText;
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI batteryText;
    public TextMeshProUGUI taskText;
    public Slider batterySlider;
    public Image statusIndicator;
    public Button emergencyStopButton;
    
    [Header("Task Progress")]
    public Slider taskProgressSlider;
    public TextMeshProUGUI taskProgressText;
    
    [Header("Robot Control")]
    public Button startTaskButton;
    public Button pauseTaskButton;
    public Button resetButton;
    
    [Header("Data Visualization")]
    public List<GameObject> sensorVisualizations;
    
    private string robotName = "Robotic Arm";
    private float batteryLevel = 100f;
    private string currentStatus = "Ready";
    private string currentTask = "Idle";
    private float taskProgress = 0f;
    private Color normalStatusColor = Color.green;
    private Color warningStatusColor = Color.yellow;
    private Color errorStatusColor = Color.red;
    
    void Start()
    {
        InitializeDashboard();
    }
    
    void InitializeDashboard()
    {
        robotNameText.text = robotName;
        UpdateStatus(currentStatus);
        UpdateBatteryLevel(batteryLevel);
        UpdateTask(currentTask);
        
        // Setup button listeners
        if (emergencyStopButton != null)
            emergencyStopButton.onClick.AddListener(EmergencyStop);
        
        if (startTaskButton != null)
            startTaskButton.onClick.AddListener(StartTask);
        
        if (pauseTaskButton != null)
            pauseTaskButton.onClick.AddListener(PauseTask);
        
        if (resetButton != null)
            resetButton.onClick.AddListener(ResetRobot);
    }
    
    public void UpdateStatus(string status)
    {
        currentStatus = status;
        statusText.text = $"Status: {status}";
        
        // Update status indicator color based on status
        if (statusIndicator != null)
        {
            switch (status.ToLower())
            {
                case "ready":
                case "idle":
                    statusIndicator.color = normalStatusColor;
                    break;
                case "warning":
                    statusIndicator.color = warningStatusColor;
                    break;
                case "error":
                case "emergency":
                    statusIndicator.color = errorStatusColor;
                    break;
                default:
                    statusIndicator.color = normalStatusColor;
                    break;
            }
        }
    }
    
    public void UpdateBatteryLevel(float level)
    {
        batteryLevel = Mathf.Clamp(level, 0f, 100f);
        batteryText.text = $"Battery: {batteryLevel:F1}%";
        
        if (batterySlider != null)
        {
            batterySlider.value = batteryLevel;
        }
        
        // Change battery color based on level
        if (batteryText != null)
        {
            if (batteryLevel < 20f)
            {
                batteryText.color = Color.red;
            }
            else if (batteryLevel < 50f)
            {
                batteryText.color = Color.yellow;
            }
            else
            {
                batteryText.color = Color.green;
            }
        }
    }
    
    public void UpdateTask(string taskName)
    {
        currentTask = taskName;
        taskText.text = $"Task: {taskName}";
    }
    
    public void UpdateTaskProgress(float progress)
    {
        taskProgress = Mathf.Clamp01(progress);
        
        if (taskProgressSlider != null)
        {
            taskProgressSlider.value = taskProgress;
        }
        
        if (taskProgressText != null)
        {
            taskProgressText.text = $"Progress: {(taskProgress * 100):F1}%";
        }
    }
    
    public void EmergencyStop()
    {
        Debug.Log("Emergency Stop Activated!");
        UpdateStatus("Emergency Stop");
        
        // In a real system, this would send an emergency stop command to the robot
        // For now, we'll just log the event
    }
    
    public void StartTask()
    {
        Debug.Log("Task Started!");
        UpdateStatus("Working");
        UpdateTaskProgress(0f);
    }
    
    public void PauseTask()
    {
        Debug.Log("Task Paused!");
        UpdateStatus("Paused");
    }
    
    public void ResetRobot()
    {
        Debug.Log("Robot Reset!");
        UpdateStatus("Ready");
        UpdateTask("Idle");
        UpdateTaskProgress(0f);
        UpdateBatteryLevel(100f);
    }
    
    // Visualization methods for sensor data
    public void ShowSensorData(string sensorName, float value, float minRange, float maxRange)
    {
        // In a real system, this would visualize sensor data on the dashboard
        Debug.Log($"Sensor {sensorName}: {value:F2} (Range: {minRange} - {maxRange})");
    }
    
    public void ShowSensorData(string sensorName, Vector3 position)
    {
        // Visualize position data
        Debug.Log($"Position Sensor {sensorName}: ({position.x:F2}, {position.y:F2}, {position.z:F2})");
    }
    
    void Update()
    {
        // Simulate battery drain over time
        if (currentStatus == "Working")
        {
            UpdateBatteryLevel(batteryLevel - 0.01f); // Drain 0.01% per frame when working
            
            // Simulate task progress
            UpdateTaskProgress(taskProgress + 0.001f);
        }
    }
}
```

## Mini-project

Create a complete Unity-based visualization system for a mobile robot that includes:

1. A photorealistic environment using HDRP
2. A robot model with accurate kinematics and dynamics
3. An HRI interface with status monitoring and control
4. Integration with sensor data visualization (LiDAR, camera, IMU)
5. VR/AR support for immersive interaction
6. Real-time performance metrics display
7. Emergency stop and safety interlocks
8. Recording and playback functionality

Your system should include:
- Complete Unity scene with HDRP lighting
- Robot controller with realistic movement
- HRI dashboard showing robot status and controls
- Sensor data visualization
- Performance optimization techniques
- Safety features implementation

## Summary

This chapter covered Unity HDRP for high-fidelity visualization and Human-Robot Interaction:

- **HDRP Setup**: Configuring Unity's High Definition Render Pipeline for photorealistic rendering
- **Robot Visualization**: Creating detailed 3D robot models with accurate kinematics
- **HRI Interfaces**: Designing intuitive user interfaces for robot control and monitoring
- **Unity-ROS Integration**: Connecting Unity with ROS 2 for bidirectional communication
- **Performance Optimization**: Techniques for maintaining real-time performance with complex scenes
- **Safety Considerations**: Implementing emergency stops and safety interlocks in HRI systems

Unity HDRP provides powerful tools for creating visually impressive simulation environments that can significantly enhance the development and testing of robotic systems.