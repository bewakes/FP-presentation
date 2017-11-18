
# coding: utf-8

# In[ ]:


## SIDE EFFECTS
# A program/function is said to have a side effect if it modifies the
# state of something outside its scope.
# EXAMPLES: display character in screen, taking user input,
#  making a network call, reading/writing to file/database, etc.

# REMEMBER!! Side effects should be avoided as much as it is possible.


# In[30]:


# IMPURE FUNCTION
# Has side effects.
# Return value does not only depends on arguments: random.random(), datetime.now(), etc.
# Can modify the arguments

# Example
curr_state = {'lives': 1, 'score': 5}
def update_score():
    global curr_state
    curr_state['score'] += 1 # SIDE EFFECT !!
    return curr_state


# In[31]:


# PURE FUNCTION
# Does not have any side effects.
# The return value just depends on its arguments.
# Does not modify the arguments.
# Depends only on the arguments.
# Just like mathematical functions like sine, cosine, log, etc.

# Example: A better version of the above impure function
def update_score(curr_state):
    new_state = dict(curr_state)
    new_state['score'] += 1
    return new_state


# In[32]:


# ANONYMOUS/LAMBDA FUNCTION

# creation of functions on the fly,
# just like we use integer/string/float.. values

# a = 3 + 5 # here, we have not defined 3 and 5

# a lambda function

double = lambda x: x*2
add = lambda x, y: x+y

print(double(3))
print(add(4,5))


# In[1]:


# HIGHER ORDER FUNCTION
# A function that takes another function as agrument 
# and/or returns another function

def operate(function, *args): # this takes 'function' as a parameter
    return function(*args)

def add(x, y): return x+y
def subtract(x, y): return x-y

print(operate(subtract, 3,4))


# In[34]:


# ANOTHER HIGHER ORDER FUNCTION
def twice_apply(f): # take in a function, returns a function that applies passed function twice
    return lambda x: f(f(x))

def subtract2(x): return subtract(x, 2)

print( twice_apply(subtract2) (9) )


# In[ ]:


## more practical
def send_request_to_google(gurl):
    response = send_request(gurl)
    ## response parse
    return parsed_response
    
def send_request_to_facebook(fburl):
    response = send_request(fburl)
    # parse response
    return parsed_response

def common_send_req(url, parse_function):
    resp = send_request(url)
    parsed= parse_function(resp)
    return parsed

def facebook_parser(resp):
    pass
def google_parser(resp):
    pass

print(common_send_req(url, facebook_parser))


# In[35]:


## MAP/FILTER/REDUCE
# Excellent tools to manipulate lists and iterators

# MAP is used when we need to apply a function to elements in a list and collect results
# Takes a function as a parameter and a iterable/list
our_list = [1,2,3,4,5,6,7,8,9]
squares = list(
    map(
        lambda x: x**2,
        our_list
       )
)
print(squares)
# LIST COMPREHENSION equivalent


# In[36]:


# FILTER
# Used to filter out elements from a list if certain condition is fulfilled
# condition is a function taking an element and returning True or False

# Let's filter even numbers out of a list
divisible_by_2 = lambda x: x%2 == 0 # our condition

our_list = [1,2,3,4,5,6,7,8,9]
evens = list(
    filter(divisible_by_2, our_list)
)
print(evens)

# LIST COMPREHENSION equivalent


# In[2]:


# REDUCE
# Ah!! another excellent tool
# Used when we need to apply a computation to the whole list and get a result

# example: Sum of elements in a list
from functools import reduce

our_list = [1,2,3,4,5,6,7,8,9,20, 100, 34, 87, 99]
list_sum = reduce(lambda x, y: x+y, our_list)
print('list sum', list_sum)

# another example: find out maximum element
# trivially we do
mx = our_list[0]
for x in our_list[1:]:
    if x>mx: mx=x
print("Trivial max:", mx)

## Reduce to the rescue!!
mx_func = lambda x, y: x if x>y else y

mx = reduce(mx_func, our_list)
print('Reduced max:', mx)


# In[11]:


## COMPOSITION
# Passing the return value of a function as an argument to the other
# Just like in mathematics we used to do FoG(x) = F(G(x))
from functools import reduce

def compose2(f1, f2):
    """Compose two functions"""
    return lambda *args: f1(f2(*args))

def compose(*functions):
    """Compose all functions passed as parameters"""
    return reduce(compose2, functions)

def add1(x): return 1+x
def sq(x): return x*x
def cube(x): return x*x*x

c = compose(sq, add1, cube) # first cube, then add 1 and then square

# NOW, c is another function which is returned by compose2.
print(c(2)) # => sq(add1(cube(2))) => sq(add1(9)) => sq(9) => 81, TADA!!


# In[15]:


## PARTIAL APPLICATION
# Applying a function to only a few of its arguments

def add(x, y): return x+y

def add10(x): return add(10, x)
def add30(x): return add(30, x)

print('add 3 and 4:', add(3,4))
print('add 10 to 33:', add10(33))
print('add 30 to 1:', add30(1))


# In[17]:


## CURRYING
# The process of converting a function taking multiple arguments to the one taking one argument

## NOTE: It has nothing to do with the curry that we eat/cook.
## Named after matematician Haskell Curry

def curry2(f): # curry a 2-ary function, i.e function taking two arguments
    def f1(x):
        def f2(y):
            return f(x, y)
        return f2
    return f1

cadd = curry2(add) # because add takes two parameters
print(cadd(3)) # it's a function

add3 = cadd(3)

print(add3(9))
# is same as
print(cadd(3)(9))


# In[19]:


# But we want to be able to curry any arguments function,
# and need to be able to call it with mutiple/all arguments too

# because, cadd(3,4) would give error
#  "takes 1 positional argument but 2 were given"

# advanced curry
def curry(func):
    """
    Curry a function.
    Result: we can then use function with one, multiple or all arguments
    """
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return (lambda *args2, **kwargs2:
                curried(*(args + args2), **dict(kwargs, **kwargs2)))
    return curried

cadd = curry(add)
# now we can do cadd(3)(4), or cadd(3,4): no errors
print(cadd(3)(4))
print(cadd(3,4))


# In[27]:


## RECURSIONS
## One of the most beautiful things that exists!!

def fib(n):
    if n<=1:return 1
    return fib(n-1) + fib(n-2)
print('fib 5: ', fib(5))

# factorial:
def fac_ugly(n):
    a = 1
    for x in range(2, n+1):
        a*=x
    return a
print('fac_ugly: ', fac_ugly(5))

# a bit better
def fac_better(n):
    return reduce(lambda x, y:x*y, range(1,n+1))
print('fac_better: ', fac_better(5))

# Best
def fac(n):
    if n<=1: return 1
    return n*fac(n-1)
print('best fac:', fac(5))


# In[29]:


## QUICK SORT:
def quick_sort(unsorted):
    if not unsorted: return []
    less = list(filter(lambda x: x< unsorted[0], unsorted[1:]))
    more = list(filter(lambda x: x>= unsorted[0], unsorted[1:]))
    return quick_sort(less) + [unsorted[0]] + quick_sort(more)

our_list = [9,3,1,6,4,8,7,12,32,43,43,23,7,0,21,15]
print(quick_sort(our_list))


# In[39]:


### LAZY LISTS
from itertools import islice

class InfList:
    def __init__(self, gen):
        self._gen = gen

    def __getitem__(self, val):
        gen = self._gen()
        l = []
        if not isinstance(val, slice):
            for x in range(val):
                v = next(gen)
            return v
        else:
            end = val.stop or 0
            start = val.start or 0
            step = val.step or 1
            l = list(islice(gen, end))
            ret_l = []
            for x in range(start, end, step):
                ret_l.append(l[x])
            return ret_l

def infnums():
    c = 0
    while True:
        yield c
        c+=1

i = InfList(infnums)

print(i[10])
print(i[:6])
print(len(i[7:90]))


# In[51]:


### LET'S convert some csv to list of dictionaries
csvstr = """name,age
bibek,22
bidhan,19
sujan,21
biplov,22
kshitiz,17"""
### WE want something like this:
##  data = [{'name': 'bibek', 'age': '22'}, ...]


# In[6]:


## NON funcional, trivial way
data = []
splitted = csvstr.split()
header = splitted[0]
keys = header.split(',')
for line in splitted[1:]:
    record = {}
    for i, y in enumerate(line.split(',')):
        record[keys[i]] = y
    data.append(record)
print(data)


# In[52]:


## Now, our very elegant functional approach
from functools import partial

def split(separator, string): return string.split(separator)
## nothing much, just making map, dict and zip compatible with the curry function I wrote
def m(f, iterable): return map(f, iterable)
def z(it1, it2): return zip(it1, it2)
def d(*args): return dict(*args)

c_zip, c_map, c_dict = curry(z), curry(m), curry(d) # CURRIED FUNCTIONS

# curry the split function
csplit = curry(split)
split_newline = csplit('\n')
split_comma = csplit(',')

dict_from_key_vals = compose(c_dict, c_zip)
csv_to_list = compose(c_map(split_comma), split_newline)

our_list = csv_to_list(csvstr)
headers = next(our_list)

data = map(partial(dict_from_key_vals, headers), our_list) # or we can use c_map

print(list(data))


# In[ ]:


### Some references from Daniel Kirsch's talk: https://www.youtube.com/watch?v=r2eZ7lhqzNE
### Functional programming Jargon
### and many other resources

