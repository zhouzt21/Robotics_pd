from typing import Any
from robotics.planner import MobileAgent
from robotics.planner.skill import SkillConfig, Skill
from robotics.ros.playground.subscriber import NavCmdVelSubscriber, SubscriberConfig


class CmdVelSkillConfig(SkillConfig):
    name: str = 'cmd_vel_skill'

class CmdVelSkill(Skill):
    agent: MobileAgent
    def reset(self, agent, obs, **kwargs):
        super().reset(agent, obs, **kwargs)
        self.subscriber = NavCmdVelSubscriber(SubscriberConfig(channel='/cmd_vel'))

    def act(self, obs, **kwargs) -> Any:
        if self.subscriber.poll():
            from geometry_msgs.msg import Twist
            print('start')
            x, y, rot = None, None, None
            while self.subscriber.poll():
                print('----')
                cmd_vel: Twist = self.subscriber.recv()
                x = cmd_vel.linear.x
                y = cmd_vel.linear.y
                rot = cmd_vel.angular.z
            return self.agent.set_base_move([x, y, rot])
        else:
            return lambda x: x

    def should_terminate(self, obs, **kwargs):
        return False