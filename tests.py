def double(x):
    x=x*2
    return x


class person:
    def __init__(self, x, y):
        self.x = x
        self.y = y




X=[person(2,5),person(10,6)]

X=(map(lambda x:double(x.x),X))

print(X)


for x in X:
    print(x)
