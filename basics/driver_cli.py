import analysis

def main():
    data = analysis.get_data()
    while True:
        s = input("Enter a key. Enter 'q' to quit: ")
        if s == 'q' or s == 'Q':
            print("Quitting...")
            break
        print("The key is", s)

if __name__ == '__main__':
    main()