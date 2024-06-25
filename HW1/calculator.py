class Calculator:
    def __init__(self, num1, num2):    # constructor
        self.num1 = num1
        self.num2 = num2
    def power(self):
        return self.num1 ** self.num2
    def addPower(self, num3):
        return (self.num1 + num3) ** self.num2
    