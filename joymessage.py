try:
    import rospy
    from sensor_msgs.msg import Joy
except ModuleNotFoundError:
    pass
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auto_aim import PoleAim

mapping = {
    "A": 0,
    "left": 6,  # 1
    "right": 6,  # -1
    "LB": 4,
    "RB": 5,
    "test_left": 6,
    "test_right": 7,
}


class JoyMessageBase:
    def __init__(self, aim: 'PoleAim'):
        self.aim = aim

    def A(self):
        targets = self.aim.process(self.aim.result)
        a = self.aim.now #???error
        for i in targets:
            if i[6] == a:
                self.aim.aim(i)

    def LB(self):
        self.aim.left()

    def RB(self):
        self.aim.right()

    def test_left(self):
        print("test left")

    def test_right(self):
        print("test right")

    def left(self):
        self.aim.servo.step(1)

    def right(self):
        self.aim.servo.step(-1)

    def run(self):
        pass


class RosJoyMessage(JoyMessageBase):
    def __init__(self, aim):
        super().__init__(aim)
        rospy.Subscriber("joy", Joy, self.joy_message)

    def joy_message(self, data):
        # print(data.buttons)
        # print(data.axes)
        if data.buttons[mapping["A"]] == 1:
            self.A()
        if data.buttons[mapping["LB"]] == 1:
            self.LB()
        if data.buttons[mapping["RB"]] == 1:
            self.RB()
        if data.buttons[mapping["test_left"]] == 1:
            self.test_left()
        if data.buttons[mapping["test_right"]] == 1:
            self.test_right()
        if data.axes[mapping["left"]] > 0:
            self.left()
        if data.axes[mapping["right"]] < 0:
            self.right()

    def run(self):
        threading.Thread(target=self.aim.detecting, name="detecting", daemon=True).start()
        rospy.spin()


class InputJoyMessage(JoyMessageBase):
    def __init__(self, aim: 'PoleAim'):
        super().__init__(aim)

    def run(self):
        threading.Thread(target=self.aim.detecting, name="detecting", daemon=True).start()
        while True:
            a = input()
            if a == "LB":
                self.LB()
            elif a == "RB":
                self.RB()
            elif a == "a":
                self.A()
            elif a == "end":
                break


if __name__ == "__main__":
    joy = RosJoyMessage()
    joy.run()
