class Result:
    val: int


class Sum():
    def execute(self, a, b):
        result = Result()
        result.val = a + b
        return result

class Messege():
    def send_messege(self, messege:int):
        print(messege)

if __name__ == "__main__":
    sum = Sum()
    print(sum.execute(5, 10).val)