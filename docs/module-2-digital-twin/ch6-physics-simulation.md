---
title: Ch6 - Physics Simulation Fundamentals
module: 2
chapter: 6
sidebar_label: Ch6: Physics Simulation Fundamentals
description: Understanding the principles of physics simulation for robotic systems
tags: [simulation, physics, robotics, dynamics, kinematics, gazebo]
difficulty: intermediate
estimated_duration: 75
---

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Physics Simulation Fundamentals

## Learning Outcomes
- Understand the core principles of physics simulation
- Explain the differences between kinematic and dynamic simulation
- Identify the mathematical models underlying physics simulation
- Understand the concepts of collision detection and response
- Recognize the importance of simulation in robotics development
- Apply physics simulation concepts to robotic systems
- Evaluate the trade-offs between simulation accuracy and computational efficiency

## Theory

### Introduction to Physics Simulation

Physics simulation is a computational method for approximating the behavior of physical systems. In robotics, physics simulation is crucial for testing algorithms, validating designs, and generating training data for AI systems before deploying on real hardware.

<MermaidDiagram chart={`
graph TD;
    A[Physics Simulation] --> B[Kinematic Simulation];
    A --> C[Dynamic Simulation];
    B --> D[Position Only];
    B --> E[No Forces];
    C --> F[Position + Velocity];
    C --> G[Forces];
    C --> H[Mass];
    C --> I[Inertia];
    
    J[Simulation Pipeline] --> K[Model Definition];
    K --> L[Physics Engine];
    L --> M[Collision Detection];
    M --> N[Force Calculation];
    N --> O[Integration];
    O --> P[State Update];
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style L fill:#2196F3,stroke:#0D47A1,color:#fff;
`} />

### Kinematics vs Dynamics

**Kinematics** studies motion without considering the forces that cause it. It focuses on geometric relationships between position, velocity, and acceleration:

- Forward kinematics: given joint angles, find end-effector position
- Inverse kinematics: given end-effector position, find required joint angles

**Dynamics** studies motion considering the forces and torques that cause it:

- Inverse dynamics: given motion, find required forces/torques
- Forward dynamics: given forces/torques, find resulting motion

### Mathematical Models

#### Rigid Body Dynamics

The motion of a rigid body is governed by Newton-Euler equations:

For translational motion:
```
F = m * a
```

For rotational motion:
```
τ = I * α
```

Where:
- F: Force vector
- m: Mass
- a: Linear acceleration
- τ: Torque vector
- I: Moment of inertia tensor
- α: Angular acceleration

#### Configuration Space (C-Space)

Configuration space is the space of all possible positions/poses for a robot. For a robot with n degrees of freedom, the configuration space is n-dimensional:

`C = R^n` for n revolute joints with no limits
`C = T^n` for n revolute joints with 2π periodicity
`C = SE(3)` for a free-floating body

### Collision Detection

Physics simulation requires efficient algorithms to detect when objects come into contact:

1. **Broad Phase**: Quickly eliminate pairs of objects that cannot possibly collide (e.g., using bounding volume hierarchies)
2. **Narrow Phase**: Perform detailed collision detection between potentially colliding objects
3. **Contact Resolution**: Calculate contact points, normals, and penetration depth

### Integration Methods

Physics simulation uses numerical integration to solve differential equations of motion. Common methods include:

- **Explicit Euler**: Simple but unstable for stiff systems
- **Semi-implicit Euler**: More stable than explicit Euler
- **Runge-Kutta (RK4)**: More accurate but computationally expensive
- **Verlet Integration**: Good for stability in position-based systems

## Step-by-Step Labs

### Lab 1: Understanding Kinematic vs Dynamic Simulation

1. **Create a simple pendulum simulation** to visualize the difference between kinematic and dynamic models:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Physical constants
g = 9.81  # gravitational acceleration
L = 1.0   # length of pendulum
m = 1.0   # mass of pendulum

def simple_pendulum_simulate(dt=0.01, duration=10, initial_angle=0.5, initial_velocity=0.0):
    """
    Simulate a simple pendulum using the small angle approximation (kinematic)
    and full nonlinear dynamics (dynamic)
    """
    # Time array
    t = np.arange(0, duration, dt)
    n_points = len(t)
    
    # Kinematic solution (small angle approximation): theta = theta_0 * cos(omega * t)
    omega = np.sqrt(g / L)
    kinematic_theta = initial_angle * np.cos(omega * t)
    kinematic_omega = -initial_angle * omega * np.sin(omega * t)
    
    # Dynamic solution using Euler integration
    dynamic_theta = np.zeros(n_points)
    dynamic_omega = np.zeros(n_points)
    dynamic_theta[0] = initial_angle
    dynamic_omega[0] = initial_velocity
    
    # Euler integration
    for i in range(1, n_points):
        # Angular acceleration: alpha = -g/L * sin(theta)
        alpha = -g/L * np.sin(dynamic_theta[i-1])
        
        # Update state
        dynamic_omega[i] = dynamic_omega[i-1] + alpha * dt
        dynamic_theta[i] = dynamic_theta[i-1] + dynamic_omega[i] * dt
    
    # Calculate positions
    kinematic_x = L * np.sin(kinematic_theta)
    kinematic_y = -L * np.cos(kinematic_theta)
    
    dynamic_x = L * np.sin(dynamic_theta)
    dynamic_y = -L * np.cos(dynamic_theta)
    
    return t, (kinematic_x, kinematic_y, kinematic_theta), (dynamic_x, dynamic_y, dynamic_theta)

# Run simulation
t, kinematic_data, dynamic_data = simple_pendulum_simulate()

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot angle vs time
ax1.plot(t, kinematic_data[2], label='Kinematic (Small Angle)', linestyle='--')
ax1.plot(t, dynamic_data[2], label='Dynamic (Nonlinear)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle (rad)')
ax1.set_title('Pendulum Angle vs Time')
ax1.legend()
ax1.grid(True)

# Plot position vs time
ax2.plot(t, kinematic_data[0], label='Kinematic X', linestyle='--')
ax2.plot(t, kinematic_data[1], label='Kinematic Y', linestyle='--')
ax2.plot(t, dynamic_data[0], label='Dynamic X')
ax2.plot(t, dynamic_data[1], label='Dynamic Y')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position (m)')
ax2.set_title('Pendulum Position vs Time')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

2. **Analyze the differences** in the simulation outputs:
   - Compare linear vs. nonlinear behavior
   - Note how the kinematic model (small angle approximation) deviates from the dynamic model with larger initial angles
   - Observe energy conservation in the dynamic model

### Lab 2: Implementing Forward Dynamics for a Simple Robot

1. **Create a simulation of a 2-DOF planar manipulator**:

```python
import numpy as np
import matplotlib.pyplot as plt

class TwoDOFManipulator:
    def __init__(self, l1=1.0, l2=0.8, m1=1.0, m2=1.0, I1=0.1, I2=0.1):
        """
        2-DOF Planar Manipulator
        l1, l2: link lengths
        m1, m2: link masses
        I1, I2: moments of inertia
        """
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.I1 = I1
        self.I2 = I2
        
    def forward_kinematics(self, q1, q2):
        """Calculate end-effector position given joint angles"""
        x1 = self.l1 * np.cos(q1)
        y1 = self.l1 * np.sin(q1)
        
        x2 = x1 + self.l2 * np.cos(q1 + q2)
        y2 = y1 + self.l2 * np.sin(q1 + q2)
        
        return x1, y1, x2, y2
    
    def jacobian(self, q1, q2):
        """Calculate the Jacobian matrix"""
        J = np.array([
            [-self.l1*np.sin(q1) - self.l2*np.sin(q1+q2), -self.l2*np.sin(q1+q2)],
            [self.l1*np.cos(q1) + self.l2*np.cos(q1+q2), self.l2*np.cos(q1+q2)]
        ])
        return J
    
    def mass_matrix(self, q1, q2):
        """Calculate the mass matrix H(q)"""
        H11 = self.I1 + self.I2 + self.m1*(self.l1/2)**2 + self.m2*((self.l1*np.cos(q1) + self.l2*np.cos(q1+q2)/2)**2 + (self.l1*np.sin(q1) + self.l2*np.sin(q1+q2)/2)**2)
        H12 = self.I2 + self.m2*(self.l1*self.l2*np.cos(q2)/2 + (self.l2/2)**2)
        H21 = H12
        H22 = self.I2 + self.m2*(self.l2/2)**2
        
        H = np.array([[H11, H12],
                      [H21, H22]])
        return H
    
    def coriolis_gravity_matrix(self, q1, q2, q1_dot, q2_dot):
        """Calculate C(q, q_dot), G(q) matrices"""
        # Simplified version - full form is more complex
        C11 = -self.m2*self.l1*self.l2*np.sin(q2)*(q2_dot + 2*q1_dot)
        C12 = -self.m2*self.l1*self.l2*np.sin(q2)*q1_dot
        C21 = self.m2*self.l1*self.l2*np.sin(q2)*q1_dot
        C22 = 0
        
        # Gravity terms
        g = 9.81
        G1 = (self.m1*self.l1/2 + self.m2*self.l1)*g*np.cos(q1) + self.m2*self.l2*g*np.cos(q1+q2)
        G2 = self.m2*self.l2*g*np.cos(q1+q2)
        
        C = np.array([[C11, C12],
                      [C21, C22]])
        G = np.array([G1, G2])
        
        return C, G

def simulate_manipulator(duration=10, dt=0.01):
    """Simulate the manipulator with applied torques"""
    manipulator = TwoDOFManipulator()
    
    # Time array
    t = np.arange(0, duration, dt)
    n_steps = len(t)
    
    # Initialize state
    q = np.array([np.pi/4, np.pi/4])  # Joint angles
    q_dot = np.array([0.0, 0.0])      # Joint velocities
    tau = np.array([0.5, 0.3])        # Applied torques
    
    # Store trajectory
    q_history = []
    pos_history = []
    
    for i in range(n_steps):
        # Store current state
        q_history.append(q.copy())
        
        # Calculate positions
        x1, y1, x2, y2 = manipulator.forward_kinematics(q[0], q[1])
        pos_history.append([x1, y1, x2, y2])
        
        # Calculate dynamics
        H = manipulator.mass_matrix(q[0], q[1])
        C, G = manipulator.coriolis_gravity_matrix(q[0], q[1], q_dot[0], q_dot[1])
        
        # Equation: H(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = tau
        q_ddot = np.linalg.solve(H, tau - C @ q_dot - G)
        
        # Integrate
        q_dot += q_ddot * dt
        q += q_dot * dt
    
    return t, np.array(q_history), np.array(pos_history)

# Run simulation
t, q_traj, pos_traj = simulate_manipulator()

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot joint angles
ax1.plot(t, q_traj[:, 0], label='Joint 1')
ax1.plot(t, q_traj[:, 1], label='Joint 2')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle (rad)')
ax1.set_title('Joint Angles vs Time')
ax1.legend()
ax1.grid(True)

# Plot end-effector trajectory
ax2.plot(pos_traj[:, 2], pos_traj[:, 3], 'b-', label='End-effector path')
ax2.plot(pos_traj[0, 2], pos_traj[0, 3], 'go', label='Start')
ax2.plot(pos_traj[-1, 2], pos_traj[-1, 3], 'ro', label='End')
ax2.set_xlabel('X Position (m)')
ax2.set_ylabel('Y Position (m)')
ax2.set_title('End-Effector Trajectory')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')

plt.tight_layout()
plt.show()
```

### Lab 3: Understanding Numerical Integration Stability

1. **Compare different integration methods** for a simple harmonic oscillator:

```python
import numpy as np
import matplotlib.pyplot as plt

def simple_harmonic_oscillator_implicit_euler(x0, v0, k, m, dt, duration):
    """Solve SHM using implicit Euler method"""
    t = np.arange(0, duration, dt)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    
    x[0] = x0
    v[0] = v0
    
    for i in range(1, len(t)):
        # Implicit Euler: x_{n+1} = x_n + dt * v_{n+1}, v_{n+1} = v_n + dt * (-k/m * x_{n+1})
        # Solve simultaneously: (x_{n+1} - x_n)/dt = v_{n+1}, (v_{n+1} - v_n)/dt = -k/m * x_{n+1}
        # Rearranging: x_{n+1} - dt*v_{n+1} = x_n, v_{n+1} + dt*(k/m)*x_{n+1} = v_n
        
        A = np.array([[1, -dt],
                      [dt*(k/m), 1]])
        b = np.array([x[i-1], v[i-1]])
        
        sol = np.linalg.solve(A, b)
        x[i] = sol[0]
        v[i] = sol[1]
        
    return t, x, v

def simple_harmonic_oscillator_explicit_euler(x0, v0, k, m, dt, duration):
    """Solve SHM using explicit Euler method"""
    t = np.arange(0, duration, dt)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    
    x[0] = x0
    v[0] = v0
    
    for i in range(1, len(t)):
        # Harmonic oscillator: dx/dt = v, dv/dt = -k/m * x
        dx_dt = v[i-1]
        dv_dt = -(k/m) * x[i-1]
        
        x[i] = x[i-1] + dx_dt * dt
        v[i] = v[i-1] + dv_dt * dt
        
    return t, x, v

def simple_harmonic_oscillator_rk4(x0, v0, k, m, dt, duration):
    """Solve SHM using Runge-Kutta 4th order method"""
    t = np.arange(0, duration, dt)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    
    x[0] = x0
    v[0] = v0
    
    for i in range(1, len(t)):
        # Define derivatives
        def dx_dt(v_val):
            return v_val
        def dv_dt(x_val):
            return -(k/m) * x_val
        
        # RK4 coefficients for x
        k1_x = dt * dx_dt(v[i-1])
        k2_x = dt * dx_dt(v[i-1] + 0.5 * dt * dv_dt(x[i-1]))
        k3_x = dt * dx_dt(v[i-1] + 0.5 * dt * dv_dt(x[i-1] + 0.5*k2_x))
        k4_x = dt * dx_dt(v[i-1] + dt * dv_dt(x[i-1] + 0.5*k3_x))
        
        # RK4 coefficients for v
        k1_v = dt * dv_dt(x[i-1])
        k2_v = dt * dv_dt(x[i-1] + 0.5*k1_x)
        k3_v = dt * dv_dt(x[i-1] + 0.5*k2_x)
        k4_v = dt * dv_dt(x[i-1] + k3_x)
        
        x[i] = x[i-1] + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        v[i] = v[i-1] + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
        
    return t, x, v

# Parameters
k = 1.0  # Spring constant
m = 1.0  # Mass
x0 = 1.0 # Initial position
v0 = 0.0 # Initial velocity
dt = 0.01
duration = 20.0

# Run simulations
t_imp, x_imp, v_imp = simple_harmonic_oscillator_implicit_euler(x0, v0, k, m, dt, duration)
t_exp, x_exp, v_exp = simple_harmonic_oscillator_explicit_euler(x0, v0, k, m, dt, duration)
t_rk4, x_rk4, v_rk4 = simple_harmonic_oscillator_rk4(x0, v0, k, m, dt, duration)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot position vs time
ax1.plot(t_imp, x_imp, label='Implicit Euler', alpha=0.7)
ax1.plot(t_exp, x_exp, label='Explicit Euler', alpha=0.7)
ax1.plot(t_rk4, x_rk4, label='RK4', alpha=0.7)
ax1.set_ylabel('Position')
ax1.set_title('Simple Harmonic Motion - Different Integration Methods')
ax1.legend()
ax1.grid(True)

# Plot velocity vs time
ax2.plot(t_imp, v_imp, label='Implicit Euler', alpha=0.7)
ax2.plot(t_exp, v_exp, label='Explicit Euler', alpha=0.7)
ax2.plot(t_rk4, v_rk4, label='RK4', alpha=0.7)
ax2.set_ylabel('Velocity')
ax2.set_xlabel('Time (s)')
ax2.legend()
ax2.grid(True)

# Phase plot (position vs velocity)
ax3.plot(x_imp, v_imp, label='Implicit Euler', alpha=0.7)
ax3.plot(x_exp, v_exp, label='Explicit Euler', alpha=0.7)
ax3.plot(x_rk4, v_rk4, label='RK4', alpha=0.7)
ax3.set_xlabel('Position')
ax3.set_ylabel('Velocity')
ax3.set_title('Phase Plot')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
```

## Runnable Code Example

Here's a complete simulation of a double pendulum that demonstrates chaotic behavior:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DoublePendulum:
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        self.m1 = m1  # Mass of first pendulum
        self.m2 = m2  # Mass of second pendulum
        self.l1 = l1  # Length of first pendulum
        self.l2 = l2  # Length of second pendulum
        self.g = g    # Gravitational acceleration
    
    def equations_of_motion(self, t, state):
        """
        Equations of motion for double pendulum
        state = [theta1, omega1, theta2, omega2]
        """
        theta1, omega1, theta2, omega2 = state
        
        # Precompute some values
        cos_delta = np.cos(theta1 - theta2)
        sin_delta = np.sin(theta1 - theta2)
        
        # Denominator for angular acceleration equations
        denom = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * cos_delta * cos_delta
        
        # Angular acceleration for theta1
        theta1_ddot = (self.m2 * self.g * self.l2 * sin_delta * cos_delta - 
                      self.m2 * self.l1 * omega1 * omega1 * sin_delta * cos_delta + 
                      self.m2 * self.g * self.l2 * np.sin(theta2) - 
                      self.m2 * self.l2 * omega2 * omega2 * sin_delta) / denom
        
        # Angular acceleration for theta2
        theta2_ddot = ((self.m1 + self.m2) * self.g * np.sin(theta1) * cos_delta - 
                      (self.m1 + self.m2) * self.l1 * omega1 * omega1 * sin_delta - 
                      (self.m1 + self.m2) * self.g * np.sin(theta2) + 
                      self.m2 * self.l2 * omega2 * omega2 * sin_delta * cos_delta) / (denom * self.l2 / self.l1)
        
        return [omega1, theta1_ddot, omega2, theta2_ddot]

def simulate_double_pendulum(initial_conditions, duration=20, dt=0.01):
    """
    Simulate double pendulum using 4th-order Runge-Kutta integration
    """
    t = np.arange(0, duration, dt)
    n_steps = len(t)
    
    # Initialize state array
    state = np.zeros((n_steps, 4))  # [theta1, omega1, theta2, omega2]
    state[0] = initial_conditions
    
    # Create instance of double pendulum
    pendulum = DoublePendulum()
    
    # RK4 integration
    for i in range(1, n_steps):
        # Get current state
        current_state = state[i-1]
        current_time = t[i-1]
        
        # RK4 coefficients
        k1 = dt * np.array(pendulum.equations_of_motion(current_time, current_state))
        k2 = dt * np.array(pendulum.equations_of_motion(current_time + dt/2, current_state + k1/2))
        k3 = dt * np.array(pendulum.equations_of_motion(current_time + dt/2, current_state + k2/2))
        k4 = dt * np.array(pendulum.equations_of_motion(current_time + dt, current_state + k3))
        
        # Update state
        state[i] = current_state + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    # Calculate positions of pendulum bobs
    x1 = pendulum.l1 * np.sin(state[:, 0])
    y1 = -pendulum.l1 * np.cos(state[:, 0])
    
    x2 = x1 + pendulum.l2 * np.sin(state[:, 2])
    y2 = y1 - pendulum.l2 * np.cos(state[:, 2])
    
    return t, state, x1, y1, x2, y2

# Simulation parameters
initial_conditions = [np.pi/2, 0, np.pi/2, 0]  # [theta1, omega1, theta2, omega2]
t, state, x1, y1, x2, y2 = simulate_double_pendulum(initial_conditions)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot angles over time
ax1.plot(t, state[:, 0], label='θ₁', linewidth=2)
ax1.plot(t, state[:, 2], label='θ₂', linewidth=2)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle (rad)')
ax1.set_title('Double Pendulum: Angles vs Time')
ax1.legend()
ax1.grid(True)

# Plot trajectory of second mass
ax2.plot(x2, y2, 'r-', alpha=0.7, label='Path of second mass')
ax2.plot([0, x1[0]], [0, y1[0]], 'bo-', markersize=10, label='Initial position', linewidth=3)
ax2.plot([x1[0], x2[0]], [y1[0], y2[0]], 'ro-', markersize=12, linewidth=3)
ax2.plot(x2[-1], y2[-1], 'rs', markersize=10, label='Final position')
ax2.set_xlabel('X Position (m)')
ax2.set_ylabel('Y Position (m)')
ax2.set_title('Double Pendulum: Trajectory of Second Mass')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')

plt.tight_layout()
plt.show()

# Animation of the double pendulum
fig, ax = plt.subplots(figsize=(8, 8))
line, = ax.plot([], [], 'o-', lw=2, markersize=8)
trace, = ax.plot([], [], 'r-', alpha=0.5, linewidth=1)

# Set up the axes
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 0.5)
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Double Pendulum Animation')
ax.grid(True)

def init():
    line.set_data([], [])
    trace.set_data([], [])
    return line, trace

def animate(i):
    # Draw the pendulum
    x_vals = [0, x1[i], x2[i]]
    y_vals = [0, y1[i], y2[i]]
    line.set_data(x_vals, y_vals)
    
    # Draw the trace of the second mass
    trace.set_data(x2[:i+1], y2[:i+1])
    
    return line, trace

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=20, blit=True, repeat=True)
plt.show()

print(f"Simulation completed for {len(t)} time steps over {t[-1]:.2f} seconds")
```

## Mini-project

Create a physics simulation of a differential drive robot that includes:

1. A model of the robot's kinematics and dynamics
2. Implementation of collision detection with walls/obstacles
3. A simple control algorithm (e.g., to follow a trajectory)
4. Visualization of the robot's motion and environment
5. Performance analysis comparing different integration methods

Your simulation should include:
- Differential drive kinematic model
- Integration of motion equations
- Obstacle representation and collision detection
- Visualization of robot trajectory
- Analysis of energy conservation or other physical properties

## Summary

This chapter introduced the fundamentals of physics simulation in robotics:

- **Kinematics vs. Dynamics**: Understanding the difference between position-based and force-based motion simulation
- **Mathematical Models**: The core equations governing rigid body motion
- **Integration Methods**: Numerical techniques for solving differential equations of motion
- **Collision Detection**: Methods for identifying when objects make contact
- **Stability Considerations**: Trade-offs between accuracy and computational efficiency

Physics simulation is essential for robotics development, allowing engineers to test algorithms, validate designs, and train AI systems in a safe, cost-effective virtual environment before deployment on real hardware.