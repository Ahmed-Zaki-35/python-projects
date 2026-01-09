import math

again = False

num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
op = input("Enter operation (+, -, *, /): ")

def calculator(a, b, operation):
    if(operation == '+'):
            return a + b;
    elif(operation == '-'):
            return a - b;
    elif(operation == '*'):
            return a * b;
    elif(operation == '/'):
        if b!= 0 :
            return a / b;
        else:
            return "you can't devide by zero";

while not again:
    result = calculator(num1, num2, op)
    print("Result: " , result)
    again_input = input("Do you want to perform another calculation? (y/n): ").strip().lower()
    if again_input == 'y':
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))
        op = input("Enter operation (+, -, *, /): ")
    else:
        again = True