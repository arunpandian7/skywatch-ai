"""
    Dirver Python File for Skywatch AI - an Face Recogniton based Attendance Monitor
"""

__author__ = "Arun Pandian R"
__version__ = "0.0.1"
__license__ = "MIT"

import argparse



def main(args):
    """ Main entry point of the app """
    if args.mode == 'transform' and args.directory != None:
        


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required argument to execute the function
    parser.add_argument("mode", help="Pass the keyword to execute")

    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-n", "--dir", action="store", dest="directory")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="SkywatchAI - version {version}".format(version=__version__))

    args = parser.parse_args()
    main(args)


