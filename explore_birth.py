import dateutil
from dateutil import parser

def main():
    test_str = "Brandy Hall 8 / 4 / 1994 & Michael sosa don ' t know the birthday"
    d = parser.isoparse(test_str)
    print(d)
    

    pass

if __name__=="__main__":
    main()

