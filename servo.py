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
        self.__move()

    def __move(self):
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


class ArduinoServo(ServoBase):
    def __init__(self, offset=-13, gear_ratio=5, pin=9, com='/dev/ttyUSB0'):
        super().__init__(offset, gear_ratio)
        self.board = pyfirmata2.Arduino(com)
        self.iter = pyfirmata2.util.Iterator(self.board)
        self.iter.start()
        self.pin = self.board.get_pin(f'd:{pin}:s')
        self.pin.write(self.deg_now)
        print(f"Servo Initialized at {com} PIN{pin}")

    def __move(self):
        self.pin.write(self.deg_now)


if __name__ == "__main__":
    servo = Servo(com="COM18")
    servo.run()
