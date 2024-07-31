from typing import TYPE_CHECKING

ROS2 = True

if TYPE_CHECKING:
    import rclpy
    ROS2 = True
else:
    try:
        import rclpy
    except Exception as e:
        import rospy
        ROS2 = False

    
if ROS2:
    import rclpy
    from rclpy.time import Time
    node: "rclpy.Node" = None # type: ignore
    use_sim_time = True

    def make_time(secs=None, nsecs=None):
        return Time(nanoseconds=int(secs * 1e9) + nsecs).to_msg()

    def get_time():
        clock = node.get_clock()
        return clock.now().to_msg()

    def init(name):
        global node, use_sim_time
        rclpy.init()
        node = rclpy.create_node(node_name=name)
        node.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, use_sim_time)])
        return node

    def spin_once():
        global node
        rclpy.spin_once(node)

    def create_publisher(topic, msg_type, queue_size):
        return node.create_publisher(msg_type, topic, queue_size)

    def create_subscriber(topic, msg_type, queue_size, callback):
        return node.create_subscription(msg_type, topic, callback, queue_size)

    def is_shutdown():
        return not rclpy.ok()

    def close():
        node.destroy_node()
        # rclpy.shutdown()

    def spin():
        rclpy.spin(node)

    def set_use_sim_time():
        #raise NotImplementedError
        # global use_sim_time
        # use_sim_time = True
        pass

    def wait_for_time():
        raise NotImplementedError

else:
    from std_msgs.msg import Time
    import rospy
    from rospy import Time

    def make_time(secs=None, nsecs=None):
        return Time(secs=secs, nsecs=nsecs)

    def get_time():
        return rospy.Time.now()

    def init(name):
        return rospy.init_node(name)

    def create_publisher(topic, msg_type, queue_size):
        return rospy.Publisher(topic, msg_type, queue_size=queue_size)

    def create_subscriber(topic, msg_type, queue_size, callback):
        return rospy.Subscriber(topic, msg_type, callback, queue_size=queue_size)

    def is_shutdown():
        return rospy.is_shutdown()

    def close():
        rospy.signal_shutdown('exit')

    def spin():
        rospy.spin()

    def set_use_sim_time():
        rospy.set_param('use_sim_time', True)

    def wait_for_time():
        from rosgraph_msgs.msg import Clock
        rospy.wait_for_message('/clock', Clock)

    def spin_once():
        pass