from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, XArm6Robotiq, XArm6AllegroLeft, XArm6AllegroRight, FloatingRobotiq2F85Gripper, XArm6PandaGripper
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PickCube-v1", max_episode_steps=80)
class PickCubeEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "xarm6_allegro_left",
        "xarm6_allegro_right",
        "floating_robotiq_2f_85_gripper",
        "xarm6_pandagripper"
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, XArm6AllegroLeft, XArm6AllegroRight, FloatingRobotiq2F85Gripper, XArm6PandaGripper]
    cube_half_size_allegro = 0.03
    cube_half_size = 0.03
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:,0] = -0.038031
            xyz[:,1] = -0.159084
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # goal_xyz = torch.zeros((b, 3))
            # goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            # goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            goal_xyz = xyz.clone()
            goal_xyz[:, 2] = xyz[:, 2] + 0.2
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            # tcp_pose=self.agent.tcp.pose.raw_pose,
            # goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                # obj_pose=self.cube.pose.raw_pose,
                obj_to_tcp_pos=self.agent.tcp.pose.p - self.cube.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        # Allegro hand robots have a different is_grasping signature
        if isinstance(self.agent, (XArm6AllegroLeft, XArm6AllegroRight)):
            is_grasped = self.agent.is_grasping(self.cube_half_size, self.cube)
        else:
            is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2) # threshold is 0.2 here

        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def staged_rewards(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

        is_grasped = info["is_grasped"]

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        place_reward *= is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        static_reward *= info["is_obj_placed"]

        return reaching_reward.mean(), is_grasped.mean(), place_reward.mean(), static_reward.mean()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel_without_gripper = self.agent.robot.get_qvel()
        if self.robot_uids == "xarm6_robotiq":
            qvel_without_gripper = qvel_without_gripper[..., :-6]
        elif self.robot_uids == "panda":
            qvel_without_gripper = qvel_without_gripper[..., :-2]
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel_without_gripper, axis=1)
        )
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward
    
    # def compute_modified_reward(self, obs: Any, action: torch.Tensor, info: Dict): # New reward designed for pickcube without grasping info
    #     cube_position_z_offseted = self.cube.pose.p.clone()
    #     cube_position_z_offseted[:, 2] += self.cube_half_size+0.01
    #     tcp_to_obj_dist = torch.linalg.norm(
    #         cube_position_z_offseted - self.agent.tcp.pose.p, axis=1
    #     )
    #     reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
    #     reward = reaching_reward

    #     is_grasped = info["is_grasped"]/2
    #     reward += is_grasped

    #     obj_to_goal_dist = torch.linalg.norm(
    #         self.goal_site.pose.p - self.cube.pose.p, axis=1
    #     )
    #     place_reward = 2*(1 - torch.tanh(1.2 * obj_to_goal_dist))
    #     # if is_grasped >= 0.5:
    #     #     reward += place_reward
    #     reward += torch.where(is_grasped >=1, place_reward, torch.zeros_like(place_reward))
    #     qvel_without_gripper = self.agent.robot.get_qvel()
    #     if self.robot_uids == "xarm6_robotiq":
    #         qvel_without_gripper = qvel_without_gripper[..., :-6]
    #     elif self.robot_uids == "panda":
    #         qvel_without_gripper = qvel_without_gripper[..., :-2]
    #     static_reward = 1 - torch.tanh(
    #         5 * torch.linalg.norm(qvel_without_gripper, axis=1)
    #     )
    #     reward += static_reward * info["is_obj_placed"]

    #     object_grabbing_closeness = self.agent.object_reward(self.cube)
    #     # if tcp_to_obj_dist < self.cube_half_size*np.sqrt(2) + 0.01:
    #     #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,0])
    #     #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,1])
    #     #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,2])
    #     #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,3])
        
    #     # the below reward encourages pressing the cube with the gripper
    #     mask = tcp_to_obj_dist < (self.cube_half_size * np.sqrt(2) + 0.01)
    #     reward += mask * (1 - torch.tanh(5 * object_grabbing_closeness[..., 0]))
    #     reward += mask * (1 - torch.tanh(5 * object_grabbing_closeness[..., 1]))
    #     reward += mask * (1 - torch.tanh(5 * object_grabbing_closeness[..., 2]))
    #     reward += mask * (1 - torch.tanh(5 * object_grabbing_closeness[..., 3]))
        
    #     reward[info["success"]] = 5

    #     joint_pos = torch.tensor(self.agent.robot.get_qpos(), dtype=torch.float32)
    #     joint_5_pos = joint_pos[..., 4]
    #     # reward += torch.where(joint_5_pos < -0.75, 0.5, -0.5)
    #     reward += 1 / (1 + torch.exp(5.8 * (joint_5_pos + 1))) - 1/(1 + torch.exp(5.8 * (-joint_5_pos + 1))) # 0.947 at -1.5 joint value, and 0.19 at -0.75 value. Check desmos for its graph

    #     # joint_6_pos = joint_pos[..., 5]
    #     # reward += (1 - torch.tanh(torch.abs(8*joint_6_pos)-2))/4 # to encourage the wrist to be close to 0


    #     return reward

    def compute_modified_reward(self, obs: Any, action: torch.Tensor, info: Dict): # staging the previous reward designed for pickcube without grasping info

        joint_pos = torch.tensor(self.agent.robot.get_qpos(), dtype=torch.float32)
        joint_5_pos = joint_pos[..., 4]
        # reward += torch.where(joint_5_pos < -0.75, 0.5, -0.5)
        reward = 1 / (1 + torch.exp(5.8 * (joint_5_pos + 1))) - 1/(1 + torch.exp(5.8 * (-joint_5_pos + 1))) # 0.947 at -1.5 joint value, and 0.19 at -0.75 value. Check desmos for its graph
        reward = 1 / (1 + torch.exp(5.8 * (joint_5_pos + 1))) # 0.947 at -1.5 joint value, and 0.19 at -0.75 value. Check desmos for its graph
        
        mask_joint_pos = joint_pos[..., 4] < -1.25
        cube_position_z_offseted = self.cube.pose.p.clone()
        cube_position_z_offseted[:, 2] += self.cube_half_size+0.01
        tcp_to_obj_dist = torch.linalg.norm(
            cube_position_z_offseted - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 + 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward[mask_joint_pos] = reaching_reward[mask_joint_pos]

        mask_reached = tcp_to_obj_dist < (self.cube_half_size * np.sqrt(2) + 0.01)
        object_grabbing_closeness = self.agent.object_reward(self.cube)
        
        # mask_thumb_close = object_grabbing_closeness[..., 0] < self.cube_half_size * np.sqrt(1.25)+ 0.013 
        thumb_reward = (1 - torch.tanh(10 * object_grabbing_closeness[..., 0]))/2
        finger1_reward = (1 - torch.tanh(10 * object_grabbing_closeness[..., 1]))/2
        finger2_reward = (1 - torch.tanh(10 * object_grabbing_closeness[..., 2]))/2
        finger3_reward = (1 - torch.tanh(10 * object_grabbing_closeness[..., 3]))/2

        # reward[mask_reached] = (2 + (1 - torch.tanh(5 * object_grabbing_closeness[..., 0])))[mask_reached]
        # reward[mask_thumb_close] = (3 + (finger1_reward + finger2_reward + finger3_reward))[mask_thumb_close]
        reward[mask_reached] = (2 + (thumb_reward + finger1_reward + finger2_reward + finger3_reward))[mask_reached]
        
        is_grasped = info["is_grasped"]/2
        mask_grasp = is_grasped >= 1
        
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 4*(1 - torch.tanh(10 * obj_to_goal_dist))
        
        reward[mask_grasp] = (4 + place_reward)[mask_grasp]


        qvel_without_gripper = self.agent.robot.get_qvel()
        if self.robot_uids == "xarm6_robotiq":
            qvel_without_gripper = qvel_without_gripper[..., :-6]
        elif self.robot_uids == "panda":
            qvel_without_gripper = qvel_without_gripper[..., :-2]
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel_without_gripper, axis=1)
        )

        reward[info["is_obj_placed"]] = (static_reward + 9)[info["is_obj_placed"]]

        # if tcp_to_obj_dist < self.cube_half_size*np.sqrt(2) + 0.01:
        #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,0])
        #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,1])
        #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,2])
        #     reward += 1 - torch.tanh(5 * object_grabbing_closeness[...,3])
        
        # the below reward encourages pressing the cube with the gripper
        
                
        reward[info["success"]] = (10+2) # 2 for success bonus


        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_modified_reward(obs=obs, action=action, info=info) / 12
        # return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    def debug(self):
        self.agent.robot.get_qpos()
        print(self.cube.pose.p)
        self.agent.debug()
