class Person:
    def __init__(self):
        self.age=10
        self.name='Balu'
    def update(self,age):
        self.age=age
    def print(self):
        print(self.name, self.age)

p1= Person()
p1.update(20)
p1.print()
Person.print(p1)