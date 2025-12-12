module.exports = {
  docs: [
    {
      type: 'category',
      label: 'AI-Native Robotics Textbook',
      items: [
        'overview',
        {
          type: 'category',
          label: 'Module 1: Robotic Nervous System',
          items: [
            'module-1-robotic-nervous-system/index',
            'module-1-robotic-nervous-system/ch1-why-ros2',
            'module-1-robotic-nervous-system/ch2-nodes-topics-services-actions',
            'module-1-robotic-nervous-system/ch3-bridging-ai-agents-rclpy',
            'module-1-robotic-nervous-system/ch4-urdf-xacro-modeling',
            'module-1-robotic-nervous-system/ch5-building-launching-packages'
          ],
        },
        {
          type: 'category',
          label: 'Module 2: Digital Twin & Simulation',
          items: [
            'module-2-digital-twin/index',
            'module-2-digital-twin/ch6-gazebo-ignition-setup-building',
            'module-2-digital-twin/ch7-simulating-sensors',
            'module-2-digital-twin/ch8-semantic-scene-understanding',
            'module-2-digital-twin/ch9-unity-hdrp-visualization',
            'module-2-digital-twin/ch10-debugging-simulations'
          ],
        },
        {
          type: 'category',
          label: 'Module 3: Isaac Platform & GPU AI',
          items: [
            'module-3-isaac-platform/index',
            'module-3-isaac-platform/ch11-nvidia-isaac-overview',
            'module-3-isaac-platform/ch12-isaac-sim-photorealistic',
            'module-3-isaac-platform/ch13-isaac-ros-accelerated-vslam',
            'module-3-isaac-platform/ch14-bipedal-locomotion-balance',
            'module-3-isaac-platform/ch15-reinforcement-learning-sim2real'
          ],
        },
        {
          type: 'category',
          label: 'Module 4: Vision-Language-Action Integration',
          items: [
            'module-4-vision-language-action/index',
            'module-4-vision-language-action/ch16-vision-language-models-robotics',
            'module-4-vision-language-action/ch17-voice-to-action-whisper',
            'module-4-vision-language-action/ch18-cognitive-task-planning-gpt4o',
            'module-4-vision-language-action/ch19-multimodal-perception-fusion',
            'module-4-vision-language-action/ch20-voice-plan-navigate-manipulate'
          ],
        },
      ],
    },
  ],
};