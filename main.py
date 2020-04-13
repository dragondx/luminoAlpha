import argparse

# handles arguments input
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', help="Select mode", required=True)
parser.add_argument('-v', '--verbose', help="verbose?", action='store_true', required=True)
args = parser.parse_args()


# select mode
def main():
    # invoke each function
    if args.mode == "demoOCR":
        pass
    elif args.mode == "demoObjRec":
        pass
    elif args.mode == "demoContext":
        pass
    elif args.mode == "demoFaceRec":
        pass
    elif args.mode == "normal":
        # create luminoManager Object to start all services
        pass

        