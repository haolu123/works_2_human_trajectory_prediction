from common.config import argparser
from common.engine import get_fp, A_star_simulation
import os,pickle

def run():
    grid_width = 10
    baseColor_fn = 'baseColor.png'
    fp_zone_fn = 'fp2_zone.png'
    fp_code_fn = 'fp2_code.png'
    
    args = argparser.parse_args()
    fp = get_fp(baseColor_fn, fp_zone_fn, fp_code_fn, grid_width)

    tempdir  = args.temp_result_dir
    flag = os.path.exists(tempdir + 'path_all.pickle')
    if args.use_save_temp_result and os.path.exists(tempdir + 'path_all.pickle'):
        with open(tempdir + 'path_all.pickle', 'rb') as f:
            path_all = pickle.load(f)
    else:
        path_all = A_star_simulation(args, fp)
        if args.save_temp_result:
            with open(tempdir + 'path_all.pickle', 'wb') as f:
                pickle.dump(path_all,f)
    
    return path_all

if __name__ == "__main__":
    path = os.getcwd()
    print("current working directory: {0}".format(path))
    if path.split('/')[-1] != 'src':
        path = os.path.join(path,'src')
        os.chdir(path)
    path_all = run()
    print(path_all[0])