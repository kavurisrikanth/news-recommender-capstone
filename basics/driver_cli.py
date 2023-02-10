import analysis

QUIT = 0
USER_CHOSE_LOG_IN = 1
USER_CHOSE_SIGN_UP = 2
LOGIN_SUCCESS = 3
LOGIN_USER_NOT_FOUND = 4
LOGIN_FAIL_DO_SIGNUP = 5
SIGNUP_SUCCESS = 6

def welcome():
    print('Welcome to OneNews. Here are your options:')
    print('1. Log in')
    print('2. Sign up')
    while True:
        s = input("Enter a key. Enter 'q' to quit: ")
        if s == 'q' or s == 'Q':
            print("Quitting...")
            break
        elif s == '1':
            # TODO: Log in
            print("Logging in...")
            return 1
        elif s == '2':
            # TODO: Sign up
            print("Signing up...")
            return 2
        else:
            print("Invalid input. Please try again.")

def login(data):
    pass

def signup(data):
    pass

def home(data):
    pass

def main():
    data = analysis.get_data()
    next_step = welcome()
    if next_step == USER_CHOSE_LOG_IN:
        next_step = login(data)
        if next_step == LOGIN_SUCCESS:
            next_step = home(data)
        
    elif next_step == 2:
        next_step = signup(data)
        if next_step == 1:
            next_step = home(data)


if __name__ == '__main__':
    main()