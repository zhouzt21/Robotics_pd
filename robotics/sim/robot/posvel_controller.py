from .controller import *
import warnings

warnings.warn("This file is deprecated, use robotics/sim/robot/controller.py instead", DeprecationWarning)

class PDJointPosVelConfig(ControllerConfig):
    speed_min: float = 0
    speed_max: float = 100
    speed_scale = 0.0314
    eps: float = 0.01
    Kp: float = 1
    Ki: float = 0
    Kd: float = 0


class PID:
    def __init__(self, kp, ki, kd) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0
        self.prev_error = 0
    
    def __call__(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class PDJointPosVelController(Controller):
    """_summary_
    The PDJointPosVelController is used for controlling the robot with position and velocity.
    """
    config: PDJointPosVelConfig
    def __init__(self, config: ControllerConfig, robot: "Robot", joint_names_or_ids: Union[List[str], List[int]]) -> None:
        super().__init__(config, robot, joint_names_or_ids)
        self._drive_target = None

    def _clear(self):
        super()._clear()
        self._drive_target = None

    def get_action_space(self):
        n = len(self.joints)
        low, high = self.get_low(), self.get_high()
        return spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        n = len(self.joints)
        stiffness = np.broadcast_to(self.config.stiffness, n)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            joint.set_drive_property(stiffness[i], damping[i], force_limit=force_limit[i], mode=self.config.mode)
            joint.set_friction(friction[i])

    def get_low(self):
        return np.append(np.float32(np.broadcast_to(self.config.lower, len(self.joints))), self.config.speed_min)

    def get_high(self):
        return np.append(np.float32(np.broadcast_to(self.config.upper, len(self.joints))), self.config.speed_max)

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        speed = action[-1]
        action = action[:-1]


        if speed >= 1.:
            # we have a new control signal
            self._drive_target = action
            # print("new target", self._drive_target)
            self.speed = speed
            self.pid = PID(self.config.Kp, self.config.Ki, self.config.Kd)


    def _before_simulation_step(self, *args, **kwargs):
        def get_speed(pos_error, pid: PID):
            qvel_goal = pid(pos_error * self._control_freq)
            qvel_limit = self.speed * self.config.speed_scale
            qvel_goal = np.clip(qvel_goal, -qvel_limit, qvel_limit)
            return qvel_goal

        if self._drive_target is not None:
            errors = self._drive_target - self.qpos
            new_vel = get_speed(errors, self.pid)

            # for i, joint in enumerate(self.joints):
            #     joint.set_drive_velocity_target(new_vel[i])
            self.robot.engine.set_joint_velocity_target(self.articulation, self.joints, self.joint_indices, new_vel)
        
    def _action_from_delta_qpos(self, delta_qpos):
        return delta_qpos

    def drive_target(self, target):
        import logging
        logging.warning("PDJointPosVelController drive_target is not implemented properly")

        assert self.config.normalize_action
        delta_qpos = target - self.qpos
        qvel = delta_qpos * self._control_freq
        #qvel = self._action_from_delta_qpos(qvel)
        speed = max(min(qvel / self.config.speed_scale, self.config.speed_max), self.config.speed_min)
        action = np.append(target, speed)
        return self.inv_scale_action(action)