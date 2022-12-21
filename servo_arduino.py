import pyfirmata2

class Servo:
    def __init__(self, pin=9, offset=9):
        self.offset = offset
        self.board=pyfirmata2.Arduino('/dev/ttyUSB0')
        self.iter = pyfirmata2.util.Iterator(self.board)
        self.iter.start()
        self.pin = self.board.get_pin(f'd:{pin}:s')
        self.deg_now = 90 + self.offset
        self.pin.write(self.deg_now)
        print("Servo Initialized")

    def move(self, angle):
        self.deg_now = max(min(self.deg_now + angle, 180), 0)
        print(self.deg_now)
        self.pin.write(self.deg_now)
        
    def run(self):
        while True:
            try:
                deg = float(input())
                self.move(deg)
            except Exception:
                print("Error input")

if __name__ == "__main__":
    servo = Servo()
    servo.run()