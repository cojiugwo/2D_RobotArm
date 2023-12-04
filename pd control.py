import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt

# Robot parameters
link_lengths = [100, 80, 60]  # Adjust lengths as needed

# PD controller gains
kp = 10  # Proportional gain
kd = 5   # Derivative gain

def forward_kinematics(joint_angles, link_lengths):
    x = np.sum([link_lengths[i] * np.cos(np.sum(joint_angles[:i+1])) for i in range(len(joint_angles))])
    y = np.sum([link_lengths[i] * np.sin(np.sum(joint_angles[:i+1])) for i in range(len(joint_angles))])
    return x, y

def pd_controller(current_position, desired_position, previous_error):
    #current_x, current_y = current_position
    #desired_x, desired_y = desired_position

    error = np.array(desired_position) - np.array(current_position)
    #error = desired_position - current_position
    derivative = error - previous_error
    command = kp * error + kd * derivative
    return command, error

# Pygame initialization
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Robot Arm Visualization")

# Initial joint angles
theta1 = np.pi / 4
theta2 = np.pi / 4
theta3 = np.pi / 4

# Arrays to store simulation results
actual_trajectory = []
desired_trajectory = []

# Simulation loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Compute current end-effector position
    current_position = forward_kinematics([theta1, theta2, theta3], link_lengths)
    actual_trajectory.append(current_position)

    # Desired trajectory (you can modify this trajectory as needed)
    target_position = pygame.mouse.get_pos()
    desired_trajectory.append(target_position)

    # PD controller
    joint_command, _ = pd_controller(current_position, target_position, 0)
    print(joint_command)

    # Update joint angles using the command
    theta1 += joint_command[0] * 0.01
    theta2 += joint_command[1] * 0.01
    theta3 += joint_command[2] * 0.01

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw the robot arm
    arm_segments = []
    current_point = np.array([width / 2, height / 2])

    for i, (length, angle) in enumerate(zip(link_lengths, [theta1, theta2, theta3])):
        angle_rad = np.radians(angle)
        next_point = current_point + np.array([length * np.cos(angle_rad), length * np.sin(angle_rad)])
        arm_segments.append((current_point, next_point))
        current_point = next_point

    for segment in arm_segments:
        pygame.draw.line(screen, (0, 0, 0), segment[0], segment[1], 5)

    # Draw end effector
    pygame.draw.circle(screen, (255, 0, 0), (int(current_point[0]), int(current_point[1])), 10)

    pygame.display.flip()

# Plot results
actual_trajectory = np.array(actual_trajectory)
desired_trajectory = np.array(desired_trajectory)

plt.figure(figsize=(10, 6))
plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], label='Actual Trajectory')
plt.plot(desired_trajectory[:, 0], desired_trajectory[:, 1], label='Desired Trajectory', linestyle='--')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.title('End-Effector Trajectory')
plt.show()

pygame.quit()
sys.exit()