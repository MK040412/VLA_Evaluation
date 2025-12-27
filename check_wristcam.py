#!/usr/bin/env python3
"""
Test script to verify PandaWristCam cameras (hand_camera + base_camera).
Uses pygame for visualization.
"""

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARN] pygame not installed. Install with: pip install pygame")


def extract_camera_rgb(obs, camera_name):
    """Extract RGB image from observation sensor_data."""
    if 'sensor_data' not in obs:
        return None

    sensor_data = obs['sensor_data']
    if camera_name not in sensor_data:
        return None

    cam_data = sensor_data[camera_name]
    if not isinstance(cam_data, dict) or 'rgb' not in cam_data:
        return None

    rgb = cam_data['rgb']

    # Convert to numpy
    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()

    # Remove batch dimension
    if rgb.ndim == 4:
        rgb = rgb[0]

    # Ensure uint8
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

    return rgb


def main():
    print("=" * 60)
    print("PandaWristCam Camera Test")
    print("=" * 60)

    # Create environment with PandaWristCam
    print("\n[INFO] Creating PickCube-v1 with panda_wristcam...")
    print("[INFO] Control mode: pd_ee_delta_pose (End-Effector delta control)")
    print("       Action space: [dx, dy, dz, droll, dpitch, dyaw, gripper]")

    sensor_cfg = dict(shader_pack='default', width=384, height=384)

    env = gym.make(
        'PickCube-v1',
        obs_mode='rgb',
        control_mode='pd_ee_delta_pose',  # EE delta pose control
        render_mode='rgb_array',
        robot_uids='panda_wristcam',
        sensor_configs=sensor_cfg,
    )

    # Print action space info
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Action space shape: {env.action_space.shape}")

    obs, _ = env.reset(seed=42)

    # Check available cameras
    print("\n[INFO] Available cameras in sensor_data:")
    if 'sensor_data' in obs:
        for cam_name, cam_data in obs['sensor_data'].items():
            if isinstance(cam_data, dict) and 'rgb' in cam_data:
                rgb = cam_data['rgb']
                if torch.is_tensor(rgb):
                    shape = tuple(rgb.shape)
                else:
                    shape = rgb.shape
                print(f"  - {cam_name}: shape={shape}")

    # Extract images
    hand_img = extract_camera_rgb(obs, 'hand_camera')
    base_img = extract_camera_rgb(obs, 'base_camera')

    if hand_img is None:
        print("[ERROR] hand_camera not found!")
    else:
        print(f"\n[INFO] hand_camera: shape={hand_img.shape}, dtype={hand_img.dtype}")

    if base_img is None:
        print("[ERROR] base_camera not found!")
    else:
        print(f"[INFO] base_camera: shape={base_img.shape}, dtype={base_img.dtype}")

    if not PYGAME_AVAILABLE:
        print("\n[INFO] pygame not available. Skipping visualization.")
        env.close()
        return

    # Initialize pygame
    pygame.init()

    # Create window (side by side: hand_camera | base_camera)
    img_size = 384
    window_width = img_size * 2 + 20  # 20px gap
    window_height = img_size + 60  # 60px for labels

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("PandaWristCam Test - [Q]uit, [R]eset, Arrow keys: move")

    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    clock = pygame.time.Clock()
    running = True
    step_count = 0

    print("\n[INFO] Starting visualization...")
    print("  - Press Q or ESC to quit")
    print("  - Press R to reset")
    print("  - Arrow keys to move robot")
    print("  - Space to toggle gripper")

    gripper_state = 1.0  # Open

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    step_count = 0
                    print("[INFO] Environment reset")
                elif event.key == pygame.K_SPACE:
                    gripper_state = -1.0 if gripper_state > 0 else 1.0
                    print(f"[INFO] Gripper: {'Open' if gripper_state > 0 else 'Closed'}")

        # Get key presses for continuous control
        keys = pygame.key.get_pressed()
        action = np.zeros(7)  # pd_ee_delta_pose: pos(3) + euler(3) + gripper(1)

        move_speed = 0.1
        if keys[pygame.K_UP]:
            action[0] = move_speed  # +X
        if keys[pygame.K_DOWN]:
            action[0] = -move_speed  # -X
        if keys[pygame.K_LEFT]:
            action[1] = move_speed  # +Y
        if keys[pygame.K_RIGHT]:
            action[1] = -move_speed  # -Y
        if keys[pygame.K_w]:
            action[2] = move_speed  # +Z
        if keys[pygame.K_s]:
            action[2] = -move_speed  # -Z

        action[6] = gripper_state

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if terminated or truncated:
            success = info.get('success', False)
            print(f"[INFO] Episode ended. Success: {success}")
            obs, _ = env.reset()
            step_count = 0

        # Extract camera images
        hand_img = extract_camera_rgb(obs, 'hand_camera')
        base_img = extract_camera_rgb(obs, 'base_camera')

        # Clear screen
        screen.fill((30, 30, 30))

        # Draw hand_camera (left)
        if hand_img is not None:
            # pygame expects (width, height, channels) with RGB
            surf = pygame.surfarray.make_surface(hand_img.swapaxes(0, 1))
            screen.blit(surf, (0, 50))

        # Draw base_camera (right)
        if base_img is not None:
            surf = pygame.surfarray.make_surface(base_img.swapaxes(0, 1))
            screen.blit(surf, (img_size + 20, 50))

        # Draw labels
        hand_label = font.render("hand_camera (wrist)", True, (255, 255, 100))
        base_label = font.render("base_camera (3rd person)", True, (100, 255, 100))
        screen.blit(hand_label, (10, 10))
        screen.blit(base_label, (img_size + 30, 10))

        # Draw step count
        step_text = small_font.render(f"Step: {step_count}", True, (200, 200, 200))
        screen.blit(step_text, (10, window_height - 25))

        # Draw controls hint
        controls = small_font.render("Arrow: XY | W/S: Z | Space: Gripper | R: Reset | Q: Quit", True, (150, 150, 150))
        screen.blit(controls, (img_size - 150, window_height - 25))

        pygame.display.flip()
        clock.tick(30)  # 30 FPS

    pygame.quit()
    env.close()
    print("\n[INFO] Test complete.")


if __name__ == "__main__":
    main()
