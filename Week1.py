#List
my_list = [1,2,3,"ahmed"]
print(my_list[0]) #to access elements in list
my_list.append("mamdouh") #to add element to the end of list
print(my_list)

#Tuple
my_tuple = (1,2,3,"ahmed")  #similar to list but its content can not be changed
print(my_tuple[2]) #to access elements in tuple
tuple_length = len(my_tuple)

#Set
my_set = {1,2,3,2,1} #the set prevents repeating elements
print(my_set)  #output : {1, 2, 3}

#Dictionary
my_dict ={"name":"Ahmed","age":22} #each element consists of a key(name) and a value(Ahmed)
#to access an element's value from its key :
print(my_dict["name"]) #output : Ahmed
#to add new element we insert its key and value
my_dict["ID"]=12
print(my_dict) #output : {'name': 'Ahmed', 'age': 22, 'ID': 12}
my_dict.keys() #returns all keys names
my_dict.values() #returns all values
#we can create more data structures with dictionaries
students = {
    "student1": {"name": "Ahmed", "age": 22, "grades": {"arabic": 98, "math": 99, "science": 95}},
    "student2": {"name": "Ali", "age": 24, "grades": {"arabic": 90, "math": 94, "science": 80}},
}
#how can we access them ?
print(students["student1"]) #here we accessed the element of student1
print(students["student1"]["name"]) #here we accessed student1's name
print(students["student1"]["grades"]) #here we accessed all student1's grades
print(students["student1"]["grades"]["math"]) #here we accessed specific grade

#Conditional statements (if.. elif.. else)
a = 4
b = 5

if a > b:
    print("a greater than b")
elif b > a:
    print("a less than b")
else:
    print("a equals b")

#While loop
counter = 0
while counter <= 10:
    print(counter)
    counter+=1 #mafesh hena 7aga esmaha counter++
    if counter == 8:
        break #breaks the loop

#For loop
my_list = [1,2,3,'a','b','c']
for i in my_list:  #iterates over all elements in the list
    print(i,end = "") #output : 123abc, we use 'end' to change last element the function will print (default: end = '\n')

for i in range(0,5): #range function generates numbers in range [0,5[ , last number not included
    print(i, end=" ") #output :0 1 2 3 4

for i in range(0,10,2): #we can add step size
    print(i, end =",") #output : 0,2,4,6,8,

for i in range(10):
    if i==2:
        continue #skips this round
    print(i,end = " ") #output : 0 1 3 4 5 6 7 8 9


#Functions
def add(num1,num2):
    return num1+num2

print(add(1,2))

#what if we are passing parameters we do not know their count?
def print_names(*names):
    for name in names:
        print(name, end = " ")

print_names("Ahmed","Mamdouh","Abd EL-Ghany") #output : Ahmed Mamdouh Abd EL-Ghany