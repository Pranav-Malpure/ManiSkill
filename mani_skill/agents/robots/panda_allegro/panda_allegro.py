from copy import deepcopy
from typing import List

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import vectorize_pose



@register_agent()
class Panda_Allegro(BaseAgent):
    uid = "panda_allegro"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2_allegro.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            "link_3.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_7.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_11.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_15.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    gripper_joint_names = [
            "joint_0.0",
            "joint_1.0",
            "joint_2.0",
            "joint_3.0",
            "joint_4.0",
            "joint_5.0",
            "joint_6.0",
            "joint_7.0",
            "joint_8.0",
            "joint_9.0",
            "joint_10.0",
            "joint_11.0",
            "joint_12.0",
            "joint_13.0",
            "joint_14.0",
            "joint_15.0",
    ]
    ee_link_name = "panda_allegro_tcp" # can check the link name in the urdf line 219

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    # Below 3 values are to be revised #TODO
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    # From allegro hand
    joint_stiffness = 4e2
    joint_damping = 1e1
    joint_force_limit = 5e1

    tip_link_names = [ # From ALLEGRO HAND
        "link_15.0_tip",
        "link_3.0_tip",
        "link_7.0_tip",
        "link_11.0_tip",
    ]

    palm_link_name = "palm" # From ALLEGRO HAND

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # # PD joint position and velocity
        # arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
        #     self.arm_joint_names,
        #     None,
        #     None,
        #     self.arm_stiffness,
        #     self.arm_damping,
        #     self.arm_force_limit,
        #     normalize_action=False,
        # )
        # arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
        #     self.arm_joint_names,
        #     -0.1,
        #     0.1,
        #     self.arm_stiffness,
        #     self.arm_damping,
        #     self.arm_force_limit,
        #     use_delta=True,
        # )

        # # -------------------------------------------------------------------------- #
        # # Gripper
        # # -------------------------------------------------------------------------- #
        # # NOTE(jigu): IssacGym uses large P and D but with force limit
        # # However, tune a good force limit to have a good mimic behavior
        # gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
        #     self.gripper_joint_names,
        #     lower=-0.01,  # a trick to have force when the object is thin
        #     upper=0.04,
        #     stiffness=self.gripper_stiffness,
        #     damping=self.gripper_damping,
        #     force_limit=self.gripper_force_limit,
        # )
        # -------------------------------------------------------------------------- #
        # Arm Allegro
        # -------------------------------------------------------------------------- #
        joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            None,
            None,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            normalize_action=False,
        )
        joint_delta_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            -0.1,
            0.1,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            use_delta=True,
        )
        joint_target_delta_pos = deepcopy(joint_delta_pos)
        joint_target_delta_pos.use_target = True

        controller_configs = dict(
            # pd_joint_delta_pos=dict(
            #     arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            # ),
            # pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            # pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            # pd_ee_delta_pose=dict(
            #     arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            # ),
            # pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
            # # TODO(jigu): how to add boundaries for the following controllers
            # pd_joint_target_delta_pos=dict(
            #     arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos
            # ),
            # pd_ee_target_delta_pos=dict(
            #     arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos
            # ),
            # pd_ee_target_delta_pose=dict(
            #     arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            # ),
            # # Caution to use the following controllers
            # pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos),
            # pd_joint_pos_vel=dict(
            #     arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos
            # ),
            # pd_joint_delta_pos_vel=dict(
            #     arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos
            # ),
            pd_joint_delta_pos=joint_delta_pos,
            pd_joint_pos=joint_pos,
            pd_joint_target_delta_pos=joint_target_delta_pos,
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        # self.finger1_link = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), "panda_leftfinger"
        # )
        # self.finger2_link = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), "panda_rightfinger"
        # )
        # self.finger1pad_link = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), "panda_leftfinger_pad"
        # )
        # self.finger2pad_link = sapien_utils.get_obj_by_name(
        #     self.robot.get_links(), "panda_rightfinger_pad"
        # )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )
        self.tip_links: List[sapien.Entity] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.palm_link: sapien.Entity = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.palm_link_name
        )

    def get_proprioception(self): # Allegro hand function
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        obs.update(
            {
                "palm_pose": self.palm_pose,
                "tip_poses": self.tip_poses.reshape(-1, len(self.tip_links) * 7),
            }
        )

        return obs


    @property
    def tip_poses(self): # Allegro hand function
        """
        Get the tip pose for each of the finger, four fingers in total
        """
        tip_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-2)

    @property
    def palm_pose(self): # Allegro hand function
        """
        Get the palm pose for allegro hand
        """
        return vectorize_pose(self.palm_link.pose, device=self.device)
    

    # def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
    #     """Check if the robot is grasping an object

    #     Args:
    #         object (Actor): The object to check if the robot is grasping
    #         min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
    #         max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
    #     """
    #     l_contact_forces = self.scene.get_pairwise_contact_forces(
    #         self.finger1_link, object
    #     )
    #     r_contact_forces = self.scene.get_pairwise_contact_forces(
    #         self.finger2_link, object
    #     )
    #     lforce = torch.linalg.norm(l_contact_forces, axis=1)
    #     rforce = torch.linalg.norm(r_contact_forces, axis=1)

    #     # direction to open the gripper
    #     ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
    #     rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
    #     langle = common.compute_angle_between(ldirection, l_contact_forces)
    #     rangle = common.compute_angle_between(rdirection, r_contact_forces)
    #     lflag = torch.logical_and(
    #         lforce >= min_force, torch.rad2deg(langle) <= max_angle
    #     )
    #     rflag = torch.logical_and(
    #         rforce >= min_force, torch.rad2deg(rangle) <= max_angle
    #     )
    #     return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    # @staticmethod
    # def build_grasp_pose(approaching, closing, center):
    #     """Build a grasp pose (panda_hand_tcp)."""
    #     assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
    #     assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
    #     assert np.abs(approaching @ closing) <= 1e-3
    #     ortho = np.cross(closing, approaching)
    #     T = np.eye(4)
    #     T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
    #     T[:3, 3] = center
    #     return sapien.Pose(T)

    # sensor_configs = [
    #     CameraConfig(
    #         uid="hand_camera",
    #         p=[0.0464982, -0.0200011, 0.0360011],
    #         q=[0, 0.70710678, 0, 0.70710678],
    #         width=128,
    #         height=128,
    #         fov=1.57,
    #         near=0.01,
    #         far=100,
    #         entity_uid="panda_hand",
    #     )
    # ]
