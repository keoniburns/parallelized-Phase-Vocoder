import subprocess
import sys
import os

# Check if the script is called with the correct number of arguments


infile = sys.argv[1]
outfile1 = sys.argv[2]
outfile2 = sys.argv[3]
program1_outputs = []


# Loop to run the programs with different input and output files

command2 = ["./smb", "sounds/" + infile, "autest/" + outfile1]
result2 = subprocess.run(command2, stdout=subprocess.PIPE, text=True)
res2 = float(result2.stdout)
print("workers,procs,mpitime,seqtime,speedup")
for i in range(2, 7, 2):  # Incrementing by 2 from 2 to 6 procs
    for j in range(2, 9, 1):  # workers
        command1 = [
            "mpirun",
            "-n",
            str(j),
            "-ppn",
            str(i),
            "./mpishift",
            "sounds/" + infile,
            "auot/" + str(i) + str(j) + outfile1,
            str(i),
        ]

        # Run the programs using subprocess
        result1 = subprocess.run(command1, stdout=subprocess.PIPE, text=True)
        res1 = float(result1.stdout)

        # program1_outputs.append(result1.stdout)
        speedup = res2 / res1
        # Print CSV line
        print(f"{j},{i},{res1},{res2},{speedup}")
