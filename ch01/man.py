class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print(f"Hello, {self.name}!")

    def goodbye(self):
        print(f"Goodbye, {self.name}!") 

m = Man("John")
m.hello()
m.goodbye()