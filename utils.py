import os

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)