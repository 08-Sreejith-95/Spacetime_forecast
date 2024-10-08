import argparse

def init_args():
    
    #M4 dataset path configuration
    parser = argparse.ArgumentParser(description= 'Parser for child models of SpaceTime model')
    parser.add_argument('--root_path', '-p', type = str, default = 'Toymodel/data')  
    parser.add_argument('--category', )
    
    args = parser.parse_args()
    
    return args
    