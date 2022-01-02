from controller import Controller
import torch
import cProfile
import pstats

OPTIMIZE = False

def main():
    with torch.no_grad():
        Controller().run()  

if OPTIMIZE:
    cProfile.run('main()', "output.dat")    
    with open("output_time.txt","w") as f:
        p = pstats.Stats("output.   dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt","w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()
else:
    main()