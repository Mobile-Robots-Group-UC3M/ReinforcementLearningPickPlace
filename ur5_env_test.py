import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from collections import namedtuple
import math
import random

class UR5RobotiqEnv(gym.Env):
    def __init__(self):
        super(UR5RobotiqEnv, self).__init__()

        # Connect to PyBullet in DIRECT mode for Colab
        self.physics_client = p.connect(p.DIRECT) 
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1 / 300)
        
        self.action_space = spaces.Box(low=np.array([0.3, -0.3]), high=np.array([0.7, 0.3]), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([0.3, -0.3]), high=np.array([0.7, 0.3]), dtype=np.float64)

        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        self.tray_id = p.loadURDF("tray/tray.urdf", [0.5, 0.9, 0.6], p.getQuaternionFromEuler([0, 0, 0]))
        self.cube_id2 = p.loadURDF("cube.urdf", [0.5, 0.9, 0.3], p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.6, useFixedBase=True)
      
        self.robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        self.cube_id = None
        self.max_steps = 100
        self.current_step = 0
        self.gripper_range = [0, 0.085] 
        
        # Video recording variables
        self.video_writer = None
        self.sim_step_counter = 0
        self.cam_width, self.cam_height = 480, 360
        
        for link_id in [12, 17]:
            p.changeDynamics(self.robot.id, link_id, lateralFriction=1000.0, spinningFriction=1.0, frictionAnchor=1)

    def step_sim_and_render(self):
        """Steps the simulation and occasionally captures a frame for video."""
        p.stepSimulation()
        self.sim_step_counter += 1
        
        # Capture at 30fps (every 10th step since simulation is 300fps)
        if self.video_writer is not None and self.sim_step_counter % 10 == 0:
            cam_target_pos = [0.5, 0, 0.6]
            cam_distance = 1.1     
            cam_yaw, cam_pitch, cam_roll = 90, -45, 0
            cam_up_axis_idx = 2
            cam_fov, cam_near_plane, cam_far_plane = 60, 0.01, 100

            view_matrix = p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll, cam_up_axis_idx)
            proj_matrix = p.computeProjectionMatrixFOV(cam_fov, self.cam_width / self.cam_height, cam_near_plane, cam_far_plane)
            
            image = p.getCameraImage(self.cam_width, self.cam_height, view_matrix, proj_matrix)[2][:, :, :3]
            self.video_writer.send(np.ascontiguousarray(image))

    def draw_boundary(self, x_range, y_range, z_height):
        corners = [
            [x_range[0], y_range[0], z_height], [x_range[1], y_range[0], z_height],
            [x_range[1], y_range[1], z_height], [x_range[0], y_range[1], z_height],
        ]
        for i in range(len(corners)):
            p.addUserDebugLine(corners[i], corners[(i + 1) % len(corners)], [1, 0, 0], lineWidth=2)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.robot.orginal_position(self) # Passed self (env) so the robot can render frames
        
        x_range = np.arange(0.4, 0.7, 0.2)
        y_range = np.arange(-0.3, 0.3, 0.2)
        cube_start_pos = [np.random.choice(x_range), np.random.choice(y_range), 0.63]
        
        self.draw_boundary([0.3, 0.7], [-0.3, 0.3], 0.63)
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        if self.cube_id:
            p.resetBasePositionAndOrientation(self.cube_id, cube_start_pos, cube_start_orn)
        else:
            self.cube_id = p.loadURDF("./urdf/cube_blue.urdf", cube_start_pos, cube_start_orn)

        self.initial_cube_pos = np.array(cube_start_pos[:2])
        self.target_pos = np.array(cube_start_pos[:2])
        return self.target_pos, {}

    def gripper_close(self):
        grip_value = self.gripper_range[1]  
        while True:
            contact_point = p.getContactPoints(bodyA=self.robot.id)
            force = {}
            if len(contact_point) > 0:
                for i in contact_point:
                    link_index = i[2]
                    if force.get(link_index) is None: force[link_index] = {17: 0, 12: 0}
                    if i[3] in [12, 17]:
                        if i[9] > force[link_index][i[3]]: force[link_index][i[3]] = i[9]

            for link_index in force:
                if force[link_index][17] > 3 and force[link_index][12] > 3:
                    return True

            if grip_value <= self.gripper_range[0]: break

            grip_value -= 0.001
            self.robot.move_gripper(grip_value)

            for _ in range(60):
                self.step_sim_and_render()

        return False

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        eef_state = p.getLinkState(self.robot.id, self.robot.eef_id)
        eef_orientation = eef_state[1]

        target_pos = np.array([action[0], action[1], 0.88]) 
        self.robot.move_arm_ik(target_pos, eef_orientation)
        
        for _ in range(100):
            self.step_sim_and_render()

        eef_state = self.robot.get_current_ee_position()
        eef_position = np.array(eef_state[0])[:2]
        distance_to_target = abs(np.linalg.norm(eef_position - self.target_pos))
        
        if distance_to_target <= 0.01:
            steps_taken = self.max_steps - self.current_step
            reward = 100 + max(0, (steps_taken * 1))
            
            target_pos = np.array([action[0], action[1], 0.8]) 
            self.robot.move_arm_ik(target_pos, eef_orientation)
            for _ in range(100):
                self.step_sim_and_render()

            success = self.gripper_close()
    
            if success:
                self.lift_object_slowly(start_pos=np.array([action[0], action[1], 0.8]), end_z=1.0, eef_orientation=eef_orientation)
          
            done = True
        elif self.current_step >= self.max_steps:
            reward = -10 * distance_to_target
            done = True
        else:
            reward = -10 * distance_to_target
            done = False
 
        return self.target_pos, reward, done, False, {}

    def lift_object_slowly(self, start_pos, end_z, eef_orientation, steps=30, sim_steps_per_move=5):
        for i in range(steps):
            intermediate_z = start_pos[2] + (end_z - start_pos[2]) * (i + 1) / steps
            lift_pos = np.array([start_pos[0], start_pos[1], intermediate_z])
            self.robot.move_arm_ik(lift_pos, eef_orientation)

            for _ in range(sim_steps_per_move):
                self.step_sim_and_render()

    def close(self):
        p.disconnect()


class UR5Robotiq85:
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0, 0.085]
        self.max_velocity = 10

    def load(self):
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()
        
    def __parse_joint_info__(self):
        jointInfo = namedtuple('jointInfo', ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            controllable = info[2] != p.JOINT_FIXED
            if controllable: self.controllable_joints.append(info[0])
            self.joints.append(jointInfo(info[0], info[1].decode("utf-8"), info[2], info[8], info[9], info[10], info[11], controllable))

        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)]

    def __setup_mimic_joints__(self):
        mimic_children_names = {'right_outer_knuckle_joint': 1, 'left_inner_knuckle_joint': 1, 'right_inner_knuckle_joint': 1, 'left_inner_finger_joint': -1, 'right_inner_finger_joint': -1}
        self.mimic_parent_id = [j.id for j in self.joints if j.name == 'finger_joint'][0]
        self.mimic_child_multiplier = {j.id: mimic_children_names[j.name] for j in self.joints if j.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id, self.id, joint_id, p.JOINT_GEAR, [0, 1, 0], [0, 0, 0], [0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def move_gripper(self, open_length):
        open_length = max(self.gripper_range[0], min(open_length, self.gripper_range[1]))
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle, force=50, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

    def move_arm_ik(self, target_pos, target_orn):
        joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, target_pos, target_orn, lowerLimits=self.arm_lower_limits, upperLimits=self.arm_upper_limits, jointRanges=self.arm_joint_ranges, restPoses=self.arm_rest_poses)
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i], maxVelocity=self.max_velocity)

    def get_current_ee_position(self):
        return p.getLinkState(self.id, self.eef_id)

    def orginal_position(self, env):
        target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i])
        for _ in range(100):
            env.step_sim_and_render()
        self.move_gripper(0.085)
        for _ in range(3500):
            env.step_sim_and_render()