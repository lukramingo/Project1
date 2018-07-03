pin= 1234
print("swipe your card")
print("enter your pin number : ")
ent_pin = int(input(" "))
bal="y "
wit = "yes"
balance = 5000
if(ent_pin== pin):
print("balance enquiry or withdrawal")
if(input== bal):
print("your acnt balance is ", balance)
elif(input==wit):
withdrawal = int(input("enter amount to withdraw :"))
withdrawal = int(input("enter amount to waithdraw"))
if (balance >= withdrawal):
print("transaction successfull")
elif(ent_pin!=pin):
print("you entered incorrect pin")

else:
print("you have insufficient balance")