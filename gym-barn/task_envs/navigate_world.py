import rospy
import numpy as np
import time
import math
from gym import spaces
from openai_ros.robot_envs import turtlebot2_env
from gym.envs.registration import register
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
timestep_limit_per_episode = 100 # Can be any Value

# TODO - change register values
register(
        id='TurtleBot2Maze-v0',
        entry_point='openai_ros:task_envs.turtlebot2.turtlebot2_maze.TurtleBot2MazeEnv',
        timestep_limit=timestep_limit_per_episode,
    )

# TODO - change names
class JackalMazeEnv(turtlebot2_env.TurtleBot2Env):
    def __init__(self):
        """
        This Task Env is designed for having Jackal in some sort of maze.
        It will learn how to move around the maze without crashing.
        """
        
        ### ACTIONS ###
        # TODO - max acceleration params
        self.min_linear_vel = 0
        self.max_linear_vel = rospy.get_param('/jackal_velocity_controller/linear/x/max_velocity')

        self.min_angular_vel = 0
        self.max_angular_vel = rospy.get_param('/jackal_velocity_controller/angular/z/max_velocity')

        # Action space is (linear velocity, angular velocity) pair
        self.action_space = spaces.Box(np.array([self.min_linear_vel, self.min_angular_vel]),
                                       np.array([self.max_linear_vel, self.max_angular_vel]),
                                       dtype=np.float32)
        
        
        ### OBSERVATIONS ###
        self.precision = 0 #TODO - precision parameter
        self.move_base_precision = 0.05 #TODO - precision parameter

        # Initial speeds
        self.init_linear_speed = 0
        self.init_angular_speed = 0
        
        # Laser scan parameters
        laser_scan = self._check_laser_scan_ready()
        self.n_laser_scan_values = len(laser_scan.ranges)
        self.max_laser_value = laser_scan.range_max
        self.min_laser_value = laser_scan.range_min
        
        # Pose and goal parameters - TODO (highest values possible for position, goal)
        self.max_odom_x = 10
        self.min_odom_x = -10
        self.max_odom_y = 10
        self.min_odom_y = -5
        self.max_odom_yaw = 3.14
        self.min_odom_yaw = -3.14

        self.max_goal_x = 10
        self.min_goal_x = -10
        self.max_goal_y = 10
        self.min_goal_y = -5
        self.max_goal_yaw = 3.14
        self.min_goal_yaw = -3.14

        # Assemble observation space -- [[LaserScan list], [x, y, yaw], [goal_x, goal_y, goal_yaw]]
        high_laser = np.full((self.n_laser_scan_values), self.max_laser_value)
        low_laser = np.full((self.n_laser_scan_values), self.min_laser_value)

        high_odom = np.array([self.max_odom_x, self.max_odom_y, self.max_odom_yaw])
        low_odom = np.array([self.min_odom_x, self.min_odom_y, self.min_odom_yaw])

        high_goal = np.array([self.max_goal_x, self.max_goal_y, self.max_goal_yaw])
        low_goal = np.array([self.min_goal_x, self.min_goal_y, self.min_goal_yaw])

        high = np.concatenate([high_laser, high_odom, high_goal])
        low = np.concatenate([low_laser, low_odom, low_goal])

        self.observation_space = spaces.Box(low, high)


        ### REWARDS
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        # TODO - get reward params
        self.step_penalty = -1      # penalty for each step without reaching goal
        self.goal_reward = 50       # reward for reaching goal

        self.cumulated_steps = 0.0
        self.cumulated_reward = 0.0
        
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(JackalMazeEnv, self).__init__()

        # rospy.logdebug("")
        # self.laser_filtered_pub = rospy.Publisher('/turtlebot2/laser/scan_filtered', LaserScan, queue_size=1)

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10,
                        min_laser_distance=-1)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        
        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of Jackal.
        :param action: The action integer that set s what movement to do next.
        """
        
        # Action is (linear velocity, angular velocity) pair
        linear_velocity = action[0]
        angular_velocity = action[1]
        rospy.logdebug("Set Action ==> " + str(linear_velocity) + ", " + str(angular_velocity))

        # Check if larger than max velocity
        # TODO - check if acceleration is greater than allowed
        self.move_base(linear_velocity, angular_velocity, epsilon=self.move_base_precision, update_rate=10)

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        
        rospy.logdebug("BEFORE DISCRET _episode_done==>"+str(self._episode_done))
        
        discretized_observations = self.discretize_observation( laser_scan,
                                                                self.new_ranges
                                                                )

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("AFTER DISCRET_episode_done==>"+str(self._episode_done))
        rospy.logdebug("END Get Observation ==>")
        return discretized_observations
        

    def _is_done(self, observations):
        
        if self._episode_done:
            rospy.logdebug("TurtleBot2 is Too Close to wall==>"+str(self._episode_done))
        else:
            rospy.logerr("TurtleBot2 is Ok ==>")

        return self._episode_done

    def _compute_reward(self, observations, done):

        if not done:
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
        else:
            reward = -1*self.end_episode_points


        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward


    # Internal TaskEnv Methods
    
    def discretize_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False
        
        discretized_ranges = []
        filtered_range = []
        #mod = len(data.ranges)/new_ranges
        mod = new_ranges
        
        max_laser_value = data.range_max
        min_laser_value = data.range_min
        
        rospy.logdebug("data=" + str(data))
        rospy.logwarn("mod=" + str(mod))
        
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or np.isinf(item):
                    #discretized_ranges.append(self.max_laser_value)
                    discretized_ranges.append(round(max_laser_value,self.dec_obs))
                elif np.isnan(item):
                    #discretized_ranges.append(self.min_laser_value)
                    discretized_ranges.append(round(min_laser_value,self.dec_obs))
                else:
                    #discretized_ranges.append(int(item))
                    discretized_ranges.append(round(item,self.dec_obs))
                    
                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logwarn("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                # We add last value appended
                filtered_range.append(discretized_ranges[-1])
            else:
                # We add value zero
                filtered_range.append(0.1)
                    
        rospy.logdebug("Size of observations, discretized_ranges==>"+str(len(discretized_ranges)))
        
        
        self.publish_filtered_laser_scan(   laser_original_data=data,
                                            new_filtered_laser_range=discretized_ranges)
        
        return discretized_ranges
        
    
    def publish_filtered_laser_scan(self, laser_original_data, new_filtered_laser_range):
        
        rospy.logdebug("new_filtered_laser_range==>"+str(new_filtered_laser_range))
        
        laser_filtered_object = LaserScan()

        h = Header()
        h.stamp = rospy.Time.now() # Note you need to call rospy.init_node() before this will work
        h.frame_id = laser_original_data.header.frame_id
        
        laser_filtered_object.header = h
        laser_filtered_object.angle_min = laser_original_data.angle_min
        laser_filtered_object.angle_max = laser_original_data.angle_max
        
        new_angle_incr = abs(laser_original_data.angle_max - laser_original_data.angle_min) / len(new_filtered_laser_range)
        
        #laser_filtered_object.angle_increment = laser_original_data.angle_increment
        laser_filtered_object.angle_increment = new_angle_incr
        laser_filtered_object.time_increment = laser_original_data.time_increment
        laser_filtered_object.scan_time = laser_original_data.scan_time
        laser_filtered_object.range_min = laser_original_data.range_min
        laser_filtered_object.range_max = laser_original_data.range_max
        
        laser_filtered_object.ranges = []
        laser_filtered_object.intensities = []
        for item in new_filtered_laser_range:
            if item == 0.0:
                laser_distance = 0.1
            else:
                laser_distance = item
            laser_filtered_object.ranges.append(laser_distance)
            laser_filtered_object.intensities.append(item)
        
        
        self.laser_filtered_pub.publish(laser_filtered_object)