#!/usr/bin/env python
#
# @MrRobotoW at The RoboVerse
# robert.wagoner@gmail.com
#
# derived from @genesis-team go2_eval.py & go2_train.py
# https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/locomotion.html
#
# Joystick training and evaluation with joystick or keyboard
# pip install pygame
# 
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

import argparse
import pickle
import time
import numpy as np
import logging
import math
import torch
import genesis as gs
import pygame 
import glob
from pathlib import Path

from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from rsl_rl.runners import OnPolicyRunner

from genesis.ext.isaacgym import terrain_utils as isaacgym_terrain_utils
from genesis.options.morphs import Terrain

script_dir = Path(__file__).resolve().parent
utils_path = (script_dir / ".." / ".." / "genesis" / "utils").resolve()
sys.path.append(str(utils_path))
from terrain import parse_terrain

VERSION = "V1.0.26"

terrain_types = [
    "flat_terrain",                # 0
    "random_uniform_terrain",      # 1
    "pyramid_sloped_terrain",      # 2
    "discrete_obstacles_terrain",  # 3
    "wave_terrain",                # 4
    "pyramid_stairs_terrain",      # 5
    "fractal_terrain",             # 6
    "sloped_terrain",              # 7
    "stepping_stones_terrain",     # 8
    "stairs_terrain"               # 9 
] 

# -----------------------------------------------------------------------------
# Configuration Functions
# -----------------------------------------------------------------------------

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100, 
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }
    return train_cfg_dict

def get_cfgs(selected_directions):
    env_cfg = {
        "num_actions": 12,
        # Default joint angles [rad]
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5 if 'forward' in selected_directions else 0,
                            -0.5 if 'reverse' in selected_directions else 0],
        "lin_vel_y_range": [0.5 if 'right' in selected_directions else 0,
                            -0.5 if 'left' in selected_directions else 0],
        "ang_vel_range": [0.5 if 'rotate_right' in selected_directions else 0,
                          -0.5 if 'rotate_left' in selected_directions else 0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def simulate_joystick():
    """
    Simulate a random joystick input.
    """
    movements = [
        [0.5, 0, 0],       # Forward
        [-0.5, 0, 0],      # Reverse
        [0, 0.5, 0],       # Right
        [0, -0.5, 0],      # Left
        [0.5, 0.5, 0],     # Diagonal Forward-Right
        [0.5, -0.5, 0],    # Diagonal Forward-Left
        [-0.5, 0.5, 0],    # Diagonal Reverse-Right
        [-0.5, -0.5, 0],   # Diagonal Reverse-Left
        [0, 0, 0.5],       # Rotate Right
        [0, 0, -0.5],      # Rotate Left
    ]
    return movements[np.random.randint(len(movements))]

# -----------------------------------------------------------------------------
# Environment Class (Go2Env)
# -----------------------------------------------------------------------------

class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True, device="cuda"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None  # Satisfies the runner requirement.
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # Add ground plane and robot from URDF
        # self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        terrain_type = terrain_types[1]
        print(f"\n🌍 Generating Terrain: {terrain_type}")

        terrains = Terrain(
            n_subterrains=(1, 1),
            subterrain_size=(15.0, 15.0),
            horizontal_scale=.05,
            vertical_scale=.05,
            height_field=None,
            subterrain_types=[[terrain_type]],
            randomize=False
        )

        # Generate terrain
        vmesh, mesh, height_field = parse_terrain(terrains, gs.surfaces.Plastic())

        # Add terrain to scene
        terrains.pos = (-7.5, -7.5, .1) # 1.0 = 5 
        self.terrain_entity = self.scene.add_entity(terrains)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                # pos=self.base_init_pos.cpu().numpy(),
                pos=(7.5, 17.5, 1.5),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        self.scene.build(n_envs=num_envs)

        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Initialize state buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # For logging extra info

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = (self.command_cfg["lin_vel_x_range"][1] - self.command_cfg["lin_vel_x_range"][0]) * \
                                     torch.rand(len(envs_idx), device=self.device) + self.command_cfg["lin_vel_x_range"][0]
        self.commands[envs_idx, 1] = (self.command_cfg["lin_vel_y_range"][1] - self.command_cfg["lin_vel_y_range"][0]) * \
                                     torch.rand(len(envs_idx), device=self.device) + self.command_cfg["lin_vel_y_range"][0]
        self.commands[envs_idx, 2] = (self.command_cfg["ang_vel_range"][1] - self.command_cfg["ang_vel_range"][0]) * \
                                     torch.rand(len(envs_idx), device=self.device) + self.command_cfg["ang_vel_range"][0]

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        envs_idx = (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(envs_idx)

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ---------------- Reward Functions ----------------
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

# -----------------------------------------------------------------------------
# Training Main Function
# -----------------------------------------------------------------------------

def train_main(args):
    gs.init(logging_level="warning")
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(args.directions)
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    resume_path = None
    start_iteration = 0
    if os.path.exists(log_dir):
        try:
            with open(f"{log_dir}/cfgs.pkl", "rb") as f:
                env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
            checkpoint = train_cfg['runner']['checkpoint']
            resume_path = os.path.join(log_dir, f"model_{checkpoint}.pt")
            if os.path.exists(resume_path):
                start_iteration = checkpoint
            else:
                resume_path = None

            model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
            if model_files:
                highest_model = max(int(f.split('_')[1].split('.')[0]) for f in model_files)
                print(f"Highest model saved: model_{highest_model}.pt")
                start_iteration = highest_model
                resume_path = os.path.join(log_dir, f"model_{highest_model}.pt")
            print(f"Resuming from iteration {start_iteration}")
        except FileNotFoundError:
            print("Configuration file not found. Starting from the beginning.")
            start_iteration = 0
    else:
        os.makedirs(log_dir, exist_ok=True)
        start_iteration = 0

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    if resume_path:
        runner.load(resume_path)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    logging.basicConfig(level=logging.INFO)
    # Use a fixed number of learning iterations per update.
    current_learning_iterations = 1
    start_time = time.time()

    for iteration in range(start_iteration, args.max_iterations):
        # Simulate joystick input
        joystick_input = simulate_joystick()
        env.commands[:, 0] = joystick_input[0]
        env.commands[:, 1] = joystick_input[1]
        env.commands[:, 2] = joystick_input[2]

        logging.info(f"Iteration {iteration + 1}/{args.max_iterations}")
        logging.info(f"Joystick Input - lin_vel_x: {joystick_input[0]}, lin_vel_y: {joystick_input[1]}, ang_vel: {joystick_input[2]}")
        logging.info(f"Commands - lin_vel_x: {env.commands[:, 0]}, lin_vel_y: {env.commands[:, 1]}, ang_vel: {env.commands[:, 2]}")

        runner.learn(num_learning_iterations=current_learning_iterations, init_at_random_ep_len=True)
        avg_reward = env.rew_buf.mean().item()
        logging.info(f"Average Reward: {avg_reward}")
        logging.info(f"Observations: {env.get_observations()}")

        if iteration % 500 == 0:
            runner.save(os.path.join(log_dir, f"model_{iteration}.pt"))
            train_cfg['runner']['checkpoint'] = iteration
            with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
                pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

        # Remove any incremental checkpoint files that are not multiples of 500.
        for file in glob.glob(os.path.join(log_dir, "model_*.pt")):
            try:
                iter_num = int(os.path.basename(file).split("_")[-1].split(".")[0])
                if iter_num % 500 != 0:
                    os.remove(file)
            except Exception as e:
                logging.warning(f"Error removing file {file}: {e}")

    total_elapsed_time = time.time() - start_time
    hours, rem = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

# -----------------------------------------------------------------------------
# Evaluation Main Function
# -----------------------------------------------------------------------------

# Constants for evaluation keyboard control
LINEAR_VELOCITY = 1.0
ANGULAR_VELOCITY_RIGHT = 2.0
ANGULAR_VELOCITY_LEFT = 2.0

def get_keyboard_input():
    keys = pygame.key.get_pressed()
    lin_vel_x = 0
    lin_vel_y = 0
    ang_vel = 0

    if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
        lin_vel_x = LINEAR_VELOCITY
        lin_vel_y = LINEAR_VELOCITY
    elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
        lin_vel_x = LINEAR_VELOCITY
        lin_vel_y = -LINEAR_VELOCITY
    elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
        lin_vel_x = -LINEAR_VELOCITY
        lin_vel_y = LINEAR_VELOCITY
    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
        lin_vel_x = -LINEAR_VELOCITY
        lin_vel_y = -LINEAR_VELOCITY
    elif keys[pygame.K_UP]:
        lin_vel_x = LINEAR_VELOCITY
    elif keys[pygame.K_DOWN]:
        lin_vel_x = -LINEAR_VELOCITY
    elif keys[pygame.K_RIGHT]:
        lin_vel_y = LINEAR_VELOCITY
    elif keys[pygame.K_LEFT]:
        lin_vel_y = -LINEAR_VELOCITY
    elif keys[pygame.K_q]:
        ang_vel = ANGULAR_VELOCITY_RIGHT
    elif keys[pygame.K_e]:
        ang_vel = -ANGULAR_VELOCITY_LEFT
    elif keys[pygame.K_h]:
        show_key_mappings()

    return [lin_vel_x, lin_vel_y, ang_vel]

def show_key_mappings():
    print("Key Mappings:")
    print("UP: Forward")
    print("DOWN: Reverse")
    print("RIGHT: Right")
    print("LEFT: Left")
    print("UP + RIGHT: Diagonal Forward-Right")
    print("UP + LEFT: Diagonal Forward-Left")
    print("DOWN + RIGHT: Diagonal Reverse-Right")
    print("DOWN + LEFT: Diagonal Reverse-Left")
    print("Q: Rotate Right")
    print("E: Rotate Left")
    print("H: Show this help message")

def eval_main(args):
    gs.init(logging_level="warning", backend=gs.gpu)
    log_dir = f"logs/{args.exp_name}"
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}  # Disable reward scaling during evaluation

    env_cfg["episode_length_s"] = 1e6
    env_cfg["termination_if_roll_greater_than"] = 100
    env_cfg["termination_if_pitch_greater_than"] = 100

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Joystick Simulation")

    obs, _ = env.reset()
    env.commands[:] = 0
    logging_enabled = False
    with torch.no_grad():
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            joystick_input = get_keyboard_input()
            if any(joystick_input):
                env.commands[:, 0] = joystick_input[0]
                env.commands[:, 1] = joystick_input[1]
                env.commands[:, 2] = joystick_input[2]
                if not logging_enabled:
                    print(f"Key Pressed - lin_vel_x: {joystick_input[0]}, lin_vel_y: {joystick_input[1]}, ang_vel: {joystick_input[2]}")
                    logging.getLogger().setLevel(logging.INFO)
                    logging_enabled = True
            else:
                env.commands[:, 0] = 0
                env.commands[:, 1] = 0
                env.commands[:, 2] = 0
                if logging_enabled:
                    logging.getLogger().setLevel(logging.WARNING)
                    logging_enabled = False

            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

# -----------------------------------------------------------------------------
# Main: Choose Train or Eval Mode via Subparsers
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Go2 ET Script {VERSION}")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run: train or eval")

    # Subparser for training mode
    train_parser = subparsers.add_parser("train", help="Training mode")
    train_parser.add_argument("-e", "--exp_name", type=str, default="etmove")
    train_parser.add_argument("-B", "--num_envs", type=int, default=5000)
    train_parser.add_argument("--max_iterations", type=int, default=5000, help="Number of training iterations")
    train_parser.add_argument("--show_viewer", action="store_true", help="Show GUI viewer during training")
    train_parser.add_argument("--directions", type=str, nargs='+', default=['forward', 'reverse', 'right', 'left', 'rotate_right', 'rotate_left'],
                              help="Specify directions to train")
    train_parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume training from the last checkpoint")
    train_parser.set_defaults(resume=True)
    # Removed the best-policy-related early stopping argument:
    # train_parser.add_argument("--early_stop_no_new_best", type=int, default=500,
    #                           help="Stop training if no improvement for this many iterations")
 
    # Subparser for evaluation mode
    eval_parser = subparsers.add_parser("eval", help="Evaluation mode")
    eval_parser.add_argument("-e", "--exp_name", type=str, default="et1move")
    eval_parser.add_argument("--ckpt", type=str, default="100", 
                               help="Checkpoint number to load for evaluation (e.g. '100')")
    # Removed the option to load 'best' model.

    args = parser.parse_args()
    
    logging.info(f"Starting Go2 ET Script {VERSION}")

    if args.mode == "eval":
        eval_main(args)
    else:
        train_main(args)

"""
Evaluation and Training Script for Go2Env.

Usage examples:
    Evaluation mode:
        python go2_et.py eval -e etmove --ckpt 2000
    Training mode:
        python go2_et.py train -e etmove --max_iterations 5000 --num_envs 30000
"""
