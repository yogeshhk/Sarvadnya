"""
  Simple module to generate Bot response.
"""
from random import randrange


def user_fact_res(u, c, l):
    r = "wow"
    cr = ["Great choice!", "I also like $"]
    lr = ["Good place to live!", "Beautiful place!",
          "I also like to live in $!"]
    # print(u, " ", c, " ", l)
    if c is None and l is None and u is not None:
        r = "Nice to meet you, " + u
    elif c is not None:
        random_index = randrange(0, len(cr))
        r = cr[random_index]
        r = r.replace("$", c)
    elif l is not None:
        random_index = randrange(0, len(lr))
        r = lr[random_index]
        r = r.replace("$", l)
    return r


def restaurant_fact_res(n, c, l):
    return ("Good to know!")


def find_restaurant_res(rn, loc, cus, last_location):
    if loc is None:
        output_str = "OH! Sorry I can't find any restaurant in " + last_location + ". Please teach me about resturants there."
    elif cus is None:
        output_str = "OH! Sorry I can't find any restaurant with type  of food you like in " + last_location + ". Please teach me about resturants there."
        output_str += "\n In the mean while, if you are really hungry check " + rn
    else:
        output_str = "How about " + rn + " which is a " + cus + " restaurant in " + loc
    return output_str
