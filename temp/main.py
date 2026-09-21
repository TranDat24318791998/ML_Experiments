from typing import Protocol

from base import Sum, Messege, Result

class Operation(Protocol):
    def execute(self, a, b) -> Result:...

def math_agent(calculator: Operation, a:int, b:int) -> Result:
    result = calculator.execute(a, b)
    print(f"Result: {result.val}")

math_agent(calculator=Sum(), a=12, b=13)

