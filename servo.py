import traceback

import pyfirmata2


class ServoBase:
    def __init__(self, offset=-13, gear_ratio=5):
        self.offset = offset
        self.gear_ratio = gear_ratio
        self.deg_now = 90 + self.offset

    def move(self, angle):
        angle = angle * self.gear_ratio
        self.deg_now = max(min(self.deg_now + angle, 180), 0)
        print(f"current_angle:{self.deg_now}, input_angle{angle}")
        self._move(self.deg_now)

    def _move(self, angle):
        pass

    def run(self):
        while True:
            try:
                deg = float(input())
                self.move(deg)
            except Exception:
                print(f"Error input:")
                for line in traceback.format_exc().splitlines():
                    print(line)


try:
    import rospy
    from std_msgs.msg import Int16

    def talker():
        rate = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            hello_str = "hello world %s" % rospy.get_time()
            rospy.loginfo(hello_str)
            pub.publish(hello_str)
            rate.sleep()

    class ROSServo(ServoBase):
        def __init__(self, offset=-13, gear_ratio=5):
            super().__init__(offset, gear_ratio)
            self.pub = rospy.Publisher('servo', Int16, queue_size=10)
            rospy.init_node('servo', anonymous=True)
            print(f"Servo Initialized with ROS node: 'servo'")

        def _move(self, angle):
            self.pub.publish(angle)
except ModuleNotFoundError:
    print("ROS not installed")


class ArduinoServo(ServoBase):
    def __init__(self, offset=-13, gear_ratio=5, pin=9, com='/dev/ttyUSB0'):
        super().__init__(offset, gear_ratio)
        self.board = pyfirmata2.Arduino(com)
        self.iter = pyfirmata2.util.Iterator(self.board)
        self.iter.start()
        self.pin = self.board.get_pin(f'd:{pin}:s')
        self.pin.write(self.deg_now)
        print(f"Servo Initialized at {com} PIN{pin}")

    def _move(self, angle):
        self.pin.write(angle)


if __name__ == "__main__":
    servo = ArduinoServo(com="COM15", gear_ratio=1)
    servo.run()
