#!/usr/bin/env python3

import sys, time, math, copy
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from robo_gym.utils import utils, mir100_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from cv_bridge import CvBridge
import cv2
from scipy.special import rel_entr

class QueenieEnv(gym.Env):
    """Queenie base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.

    Attributes:
        mir100 (:obj:): Robot utilities object.
        observation_space (:obj:): Environment observation space.
        action_space (:obj:): Environment action space.
        distance_threshold (float): Minimum distance (m) from target to consider it reached.
        min_target_dist (float): Minimum initial distance (m) between robot and target.
        max_vel (numpy.array): # Maximum allowed linear (m/s) and angular (rad/s) velocity.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.
        laser_len (int): Length of laser data array included in the environment state.

    """

    real_robot = False
    laser_len = 16
    relative_twist_len = 6
    in_contact_len = 1

    max_episode_steps = 150

    def __init__(self, rs_address=None, **kwargs):

        self.brige = CvBridge()
        self.mir100 = mir100_utils.Mir100()
        self.elapsed_steps = 0
        # create observation space
        self.observation_space = self._get_observation_space()
        # create action space
        self.action_space = spaces.Box(low=np.full((2), -1.0), high=np.full((2), 1.0), dtype=np.float32)
        self.seed()
        self.distance_threshold = 0.2
        self.min_target_dist = 1.0
        self.min_object_dist = 2.5
        self.laser_collision_threshold = 0.65
        # Maximum linear velocity (m/s) of MiR
        max_lin_vel = 0.2
        # Maximum angular velocity (rad/s) of MiR
        max_ang_vel = 0.3
        self.max_vel = np.array([max_lin_vel, max_ang_vel])
        self.previous_camera_image = None
        self.current_camera_image = None

        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, start_pose = None, target_pose = None, object_pose=None):
        """Environment reset.

        Args:
            start_pose (list[3] or np.array[3]): [x,y,yaw] initial robot position.
            target_pose (list[3] or np.array[3]): [x,y,yaw] target robot position.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.prev_base_reward = None
        self.prev_distance_to_handle_reward = None

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len() - 64*64*3)

        # Set Robot starting position
        if start_pose:
            assert len(start_pose)==3
        else:
            start_pose = self._get_start_pose()

        rs_state[3:6] = start_pose

        # Set target position
        if target_pose:
            assert len(target_pose)==3
        else:
            target_pose = self._get_target(start_pose)
        rs_state[0:3] = target_pose

        if object_pose:
            assert len(object_pose) == 3
        else:
            object_pose = self._get_object_pose(start_pose)
        rs_state[6:9] = object_pose

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state = rs_state.tolist())
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        return self.state

    def _reward(self, rs_state, action):
        return 0, False, {}

    def step(self, action):
        
        action = action.astype(np.float32)

        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Convert environment action to Robot Server action
        rs_action = copy.deepcopy(action)
        # Scale action
        rs_action = np.multiply(action, self.max_vel)
        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")

        # Get state from Robot Server
        rs_state = self.client.get_state_msg().state
        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state, rs_action)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        # Assign reward
        reward, done, info = self._reward(rs_state=rs_state, action=action)

        return self.state, reward, done, info

    def render(self):
        pass

    def _get_robot_server_state_len(self):
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """

        target = [0.0] * 3              # 0-3
        queenie_pose = [0.0] * 3        # 3-6
        object_pose = [0.0] * 3         # 6-9
        queenie_twist = [0.0] * 2       # 9-11
        min_distance_to_handle = [0.0]  # 11-12
        angle_to_handle = [0.0]         # 12-13
        left_finger_contact = [0.0]     # 13-14
        right_finger_contact = [0.0]    # 14-15
        palm_contact = [0.0]            # 15-16
        camera_image = [0] * 84*84*3    
        rs_state = target + queenie_pose + object_pose + queenie_twist + min_distance_to_handle \
                    + angle_to_handle + left_finger_contact + right_finger_contact + palm_contact + camera_image

        return len(rs_state)

    def _get_env_state_len(self):
        """Get length of the environment state.

        Describes the composition of the environment state and returns
        its length.

        Returns:
            int: Length of the environment state

        """
        step_count = [0]
        queenie_twist = [0.0] * 2
        # visible_handle_points = [0.0]
        min_distance_to_handle = [0.0]
        angle_to_handle = [0.0]
        camera_image = [0] * 64*64*3
        
        env_state = step_count + queenie_twist + min_distance_to_handle + angle_to_handle + camera_image

        return len(env_state)

    def _get_start_pose(self):
        """Get initial robot coordinates.

        For the real robot the initial coordinates are its current coordinates
        whereas for the simulated robot the initial coordinates are
        randomly generated.

        Returns:
            numpy.array: [x,y,yaw] robot initial coordinates.

        """

        if self.real_robot:
            # Take current robot position as start position
            start_pose = self.client.get_state_msg().state[3:6]
        else:
            # Create random starting position
            x = self.np_random.uniform(low= -1.0, high= 1.0)
            y = self.np_random.uniform(low= -1.0, high= 1.0)
            yaw = self.np_random.uniform(low= -np.pi, high= np.pi)
            start_pose = [0,0,0]

        return start_pose

    def _get_target(self, robot_coordinates):
        """Generate coordinates of the target at a minimum distance from the robot.

        Args:
            robot_coordinates (list): [x,y,yaw] coordinates of the robot.

        Returns:
            numpy.array: [x,y,yaw] coordinates of the target.

        """

        target_far_enough = False
        while not target_far_enough:
            x_t = self.np_random.uniform(low= -1.0, high= 1.0)
            y_t = self.np_random.uniform(low= -1.0, high= 1.0)
            yaw_t = 0.0
            target_dist = np.linalg.norm(np.array([x_t,y_t]) - np.array(robot_coordinates[0:2]), axis=-1)

            if target_dist >= self.min_target_dist:
                target_far_enough = True

        return [x_t,y_t,yaw_t]

    def _get_object_pose(self, object_coordinates):
        object_far_enough = False

        while not object_far_enough:
            x_t = self.np_random.uniform(low= 3.4, high= 3.5)
            y_t = self.np_random.uniform(low= -1.0, high= 1.0)
            yaw_t = self.np_random.uniform(low=0.642, high=1.642)
            # dst_to_obj = np.linalg.norm(np.array([x_t,y_t]) - np.array(object_coordinates[0:2]), axis=-1)

            # if dst_to_obj >= self.min_object_dist:
            object_far_enough = True
        
        return [x_t,y_t,yaw_t]

    def _robot_server_state_to_env_state(self, rs_state, rs_action=[0,0]):
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Convert to numpy array and remove NaN values
        rs_state = np.nan_to_num(np.array(rs_state))

        # laser = utils.downsample_list_to_len(rs_state[14:147],self.laser_len)

        img_obs = np.array(rs_state[16:], dtype=np.uint8)
        img_obs = img_obs.reshape(3, 84, 84)
        self.current_camera_image = cv2.cvtColor(copy.deepcopy(img_obs).transpose((1,2,0)), cv2.COLOR_BGR2GRAY)
        self.current_camera_image = self.current_camera_image.astype(np.float32) / 255.0

        state = {}
        state['vect_obs'] = np.concatenate([np.array([self.elapsed_steps]), rs_state[9:11] , rs_state[11:13]], dtype=np.float32)
        state['image_obs'] = img_obs


        # state = np.concatenate([state, np.array(laser)])

        return state

    def _get_observation_space(self):
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """

        # # Goal coordinates range
        # max_goal_coords = np.array([np.inf,np.pi])
        # min_goal_coords = np.array([-np.inf,-np.pi])

        # Robot velocity range tolerance

        vel_tolerance = 0.1
        # step_count 
        min_step_count = np.array([0])
        max_step_count = np.array([self.max_episode_steps + 1])

        # Robot velocity range used to determine if there is an error in the sensor readings
        max_lin_vel = 1
        min_lin_vel = -1
        max_ang_vel = 1.5
        min_ang_vel = -1.5
        max_vel = np.array([max_lin_vel,max_ang_vel])
        min_vel = np.array([min_lin_vel,min_ang_vel])

        # max_visible_handle_points = np.array([np.inf])
        # min_visible_handle_points = np.array([-np.inf])

        max_min_distance_to_handle = np.array([np.inf])
        min_min_distance_to_handle = np.array([-np.inf])

        max_angle_to_handle = np.array([np.pi])
        min_angle_to_handle = np.array([-np.pi])

        # max_laser_scan = np.full(self.laser_len, 10.1)
        # min_laser_scan = np.full(self.laser_len, 0.0)

        # Definition of environment observation_space
        max_obs = np.concatenate((max_step_count, max_vel, max_min_distance_to_handle, max_angle_to_handle))
        min_obs = np.concatenate((min_step_count, min_vel, min_min_distance_to_handle, min_angle_to_handle))

        image_obs = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
        vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

        dict_obs = spaces.Dict({
            "image_obs": image_obs,
            "vect_obs": vect_obs
        })

        return dict_obs

    def _robot_outside_of_boundary_box(self, robot_coordinates):
        """Check if robot is outside of boundary box.

        Check if the robot is outside of the boundaries defined as a box with
        its center in the origin of the map and sizes width and height.

        Args:
            robot_coordinates (list): [x,y] Cartesian coordinates of the robot.

        Returns:
            bool: True if outside of boundaries.

        """

        # Dimensions of boundary box in m, the box center corresponds to the map origin
        width = 10
        height = 10

        if np.absolute(robot_coordinates[0]) > (width/2) or \
            np.absolute(robot_coordinates[1] > (height/2)):
            return True
        else:
            return False

    def _sim_robot_collision(self, rs_state):
        """Get status of simulated collision sensor.

        Used only for simulated Robot Server.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if the robot is in collision.

        """

        if rs_state[1020] == 1:
            return True
        else:
            return False

    def _min_laser_reading_below_threshold(self, rs_state):
        """Check if any of the laser readings is below a threshold.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if any of the laser readings is below the threshold.

        """

        threshold = 0.15
        if min(rs_state[8:1020]) < threshold:
            return True
        else:
            return False
    def _check_collision_laser(self, laser_data):
        min_reading = min(laser_data)
        if min_reading < self.laser_collision_threshold and self.elapsed_steps > 30:
            return True
        return False
    
    def _has_collided(self, contact_data):
        return any(contact_data)

    def _far_from_object(self, data):
        robot_pose = data[0:3]
        object_pose = data[3:6]
        if np.linalg.norm(np.array(robot_pose) - np.array(object_pose), axis=-1) > 7:
            return True
        return False
class GraspQueenie(QueenieEnv):
    # laser_len = 133

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}
        linear_power = 0
        angular_power = 0
        

        # get the number of visible points in segmented point cloud
        # num_visible_points = rs_state[11]

        # is the handle visible:
        distance_to_handle = rs_state[11]
        is_handle_visible = 1 if distance_to_handle < 20 else 0

        # get angle angle between the robot and the handle
        angle_to_handle = rs_state[12]

        # Reward base
        base_reward = is_handle_visible * (1/(distance_to_handle + 1e-6) + (1 / (abs(angle_to_handle - np.pi/2) + 0.05))*0.1)
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        linear_power = abs(action[0] *0.10)
        angular_power = abs(action[1] *0.01)

        reward -= linear_power
        reward -= angular_power

        # negative reward for every time step elapsed
        # reward = base_reward

        # reward for keeping the image in sight
        # reward += self._calculate_difference_image_obs()*10

        # End episode if termination conditions meet
        if self._robot_or_object_outside_of_boundary_box(rs_state[3:9]) or self._far_from_object(rs_state[3:9]) or not is_handle_visible:
            reward = -100
            done = True
            info['final_status'] = 'out of boundary'

        # The episode terminates with success if the distance between the robot
        # and the handle is less than the distance threshold.
        if rs_state[15] == 1:
            reward = 100
            done = True
            info['final_status'] = 'success'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info
    
    def _robot_or_object_outside_of_boundary_box(self, poses):
        object_pose = poses[3:5]
        robot_pose = poses[0:2]
        if self._robot_outside_of_boundary_box(object_pose) or self._robot_outside_of_boundary_box(robot_pose):
            return True

    

    
class GraspQueenieSim(GraspQueenie, Simulation):
    cmd = "roslaunch queenie_robot_server sim_robot_server.launch"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        GraspQueenie.__init__(self, rs_address=self.robot_server_ip, **kwargs)

























class ExplorationQueenie(QueenieEnv):
    # laser_len = 133

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}
        linear_power = 0
        angular_power = 0
        

        # get the number of visible points in segmented point cloud
        # num_visible_points = rs_state[11]

        # is the handle visible:
        distance_to_handle = rs_state[11]
        is_handle_visible = 1 if distance_to_handle < 20 else 0

        # get angle angle between the robot and the handle
        angle_to_handle = rs_state[12]

        # Reward base
        base_reward = is_handle_visible * (1/(distance_to_handle + 1e-6) + (1 / (abs(angle_to_handle - np.pi/2) + 0.05))*0.2)
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        # negative reward for every time step elapsed
        reward = base_reward - 0.1

        # reward for keeping the image in sight
        reward += self._calculate_difference_image_obs()*10

        # End episode if termination conditions meet
        if self._robot_outside_of_boundary_box(rs_state[3:5]) or self._has_collided(rs_state[13:15]) or self._far_from_object(rs_state[3:9]):
            reward = -100
            done = True
            info['final_status'] = 'out of boundary'

        # The episode terminates with success if the distance between the robot
        # and the handle is less than the distance threshold.
        if (distance_to_handle <= 0.6 and abs(angle_to_handle - np.pi/2) < 0.6):
            reward = 100
            done = True
            info['final_status'] = 'success'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info
    
  
    def _calculate_difference_image_obs(self):
        if self.previous_camera_image is None:
            self.previous_camera_image = copy.deepcopy(self.current_camera_image)
            return 0.1
        else:
            pdf_old = cv2.calcHist([self.previous_camera_image],[0],None,[16],[0,1])
            pdf_old = pdf_old / pdf_old.sum()
            pdf_new = cv2.calcHist([self.current_camera_image],[0],None,[16],[0,1])
            pdf_new = pdf_new / pdf_new.sum()
            kl_div = rel_entr(pdf_old, pdf_new)
            kl_div[np.isinf(kl_div)] = 0
            score = kl_div.sum()
            self.previous_camera_image = copy.deepcopy(self.current_camera_image)
            return score

class ExplorationQueenieSim(ExplorationQueenie, Simulation):
    cmd = "roslaunch queenie_robot_server sim_robot_server.launch"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ExplorationQueenie.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ExplorationQueenieRob(ExplorationQueenie):
    real_robot = True

class ObstacleAvoidanceQueenie(QueenieEnv):
    laser_len = 16

    def reset(self, start_pose = None, target_pose = None):
        """Environment reset.

        Args:
            start_pose (list[3] or np.array[3]): [x,y,yaw] initial robot position.
            target_pose (list[3] or np.array[3]): [x,y,yaw] target robot position.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.prev_base_reward = None

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # Set Robot starting position
        if start_pose:
            assert len(start_pose)==3
        else:
            start_pose = self._get_start_pose()

        rs_state[3:6] = start_pose

        # Set target position
        if target_pose:
            assert len(target_pose)==3
        else:
            target_pose = self._get_target(start_pose)
        rs_state[0:3] = target_pose

        # Generate obstacles positions
        self._generate_obstacles_positions()
        rs_state[6:7] = self.sim_obstacles[0]
        rs_state[7:8] = self.sim_obstacles[1]
        rs_state[8:9] = self.sim_obstacles[2]

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state = rs_state.tolist())
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        return self.state

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}
        linear_power = 0
        angular_power = 0

        # Calculate distance to the target
        target_coords = np.array([rs_state[0], rs_state[1]])
        mir_coords = np.array([rs_state[3],rs_state[4]])
        euclidean_dist_2d = np.linalg.norm(target_coords - mir_coords, axis=-1)

        
        # Reward base
        base_reward = -50*euclidean_dist_2d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        # Power used by the motors
        linear_power = abs(action[0] *0.30)
        angular_power = abs(action[1] *0.03)
        reward-= linear_power
        reward-= angular_power

        # End episode if robot is collides with an object, if it is too close
        # to an object.
        if not self.real_robot:
            if self._sim_robot_collision(rs_state) or \
            self._min_laser_reading_below_threshold(rs_state) or \
            self._robot_close_to_sim_obstacle(rs_state):
                reward = -200.0
                done = True
                info['final_status'] = 'collision'

        if (euclidean_dist_2d < self.distance_threshold):
            reward = 100
            done = True
            info['final_status'] = 'success'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info

    def _get_start_pose(self):
        """Get initial robot coordinates.

        For the real robot the initial coordinates are its current coordinates
        whereas for the simulated robot the initial coordinates are
        randomly generated.

        Returns:
            numpy.array: [x,y,yaw] robot initial coordinates.

        """

        if self.real_robot:
            # Take current robot position as start position
            start_pose = self.client.get_state_msg().state[3:6]
        else:
            # Create random starting position
            x = self.np_random.uniform(low= -2.0, high= 2.0)
            if np.random.choice(a=[True,False]):
                y = self.np_random.uniform(low= -3.1, high= -2.1)
            else:
                y = self.np_random.uniform(low= 2.1, high= 3.1)
            yaw = self.np_random.uniform(low= -np.pi, high=np.pi)
            start_pose = [x,y,yaw]

        return start_pose

    def _get_target(self, robot_coordinates):
        """Generate coordinates of the target at a minimum distance from the robot.

        Args:
            robot_coordinates (list): [x,y,yaw] coordinates of the robot.

        Returns:
            numpy.array: [x,y,yaw] coordinates of the target.

        """

        target_far_enough = False
        while not target_far_enough:
            x_t = self.np_random.uniform(low= -2.0, high= 2.0)
            if robot_coordinates[1]>0:
                y_t = self.np_random.uniform(low= -3.1, high= -2.1)
            else:
                y_t = self.np_random.uniform(low= 2.1, high= 3.1)
            yaw_t = 0.0
            target_dist = np.linalg.norm(np.array([x_t,y_t]) - np.array(robot_coordinates[0:2]), axis=-1)
            if target_dist >= self.min_target_dist:
                target_far_enough = True

        return [x_t,y_t,yaw_t]

    def _robot_close_to_sim_obstacle(self, rs_state):
        """Check if the robot is too close to one of the obstacles in simulation.

        Check if one of the corner of the robot's base has a distance shorter
        than the safety radius from one of the simulated obstacles. Used only for
        simulated Robot Server.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if the robot is too close to an obstacle.

        """

        # Minimum distance from obstacle center
        safety_radius = 0.40

        robot_close_to_obstacle = False
        robot_corners = self.mir100.get_corners_positions(rs_state[3], rs_state[4], rs_state[5])

        for corner in robot_corners:
            for obstacle_coord in self.sim_obstacles:
                if utils.point_inside_circle(corner[0],corner[1],obstacle_coord[0],obstacle_coord[1],safety_radius):
                    robot_close_to_obstacle = True

        return robot_close_to_obstacle

    def _generate_obstacles_positions(self,):
        """Generate random positions for 3 obstacles.

        Used only for simulated Robot Server.

        """

        x_0 = self.np_random.uniform(low= -2.4, high= -1.5)
        y_0 = self.np_random.uniform(low= -1.0, high= 1.0)
        yaw_0 = self.np_random.uniform(low= -np.pi, high=np.pi)
        x_1 = self.np_random.uniform(low= -0.5, high= 0.5)
        y_1 = self.np_random.uniform(low= -1.0, high= 1.0)
        yaw_1 = self.np_random.uniform(low= -np.pi, high=np.pi)
        x_2 = self.np_random.uniform(low= 1.5, high= 2.4)
        y_2 = self.np_random.uniform(low= -1.0, high= 1.0)
        yaw_2 = self.np_random.uniform(low= -np.pi, high=np.pi)

        self.sim_obstacles = [[x_0, y_0, yaw_0],[x_1, y_1, yaw_1],[x_2, y_2, yaw_2]]

class ObstacleAvoidanceQueenieSim(ObstacleAvoidanceQueenie, Simulation):
    cmd = "roslaunch mir100_robot_server sim_robot_server.launch world_name:=lab_6x8.world"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ObstacleAvoidanceQueenie.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidanceQueenieRob(ObstacleAvoidanceQueenie):
    real_robot = True
