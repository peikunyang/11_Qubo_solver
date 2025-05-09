import numpy as np
import re

def read_pyqubo_results(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.readlines()
    energy = None
    time = None
    step = None
    for line in content:
        if "Energy" in line:
            energy = float(re.search(r"[-+]?[0-9]*\.?[0-9]+", line).group())
        elif "Time" in line:
            time = float(re.search(r"[-+]?[0-9]*\.?[0-9]+", line).group())
    return energy, time

Ene=[]
Time=[]
Step=[]
datas=['1_data_1000','2_data_6000']

for i in range (1,2,1):
  for data in datas:
    for j in range (1,6,1):
      for k in range (1,6,1):
        filename = f"../{i}_th_-{i}/{data}/1_solve/solution/{j}/pyqubo_{k}"
        energy, time = read_pyqubo_results(filename)
        Ene.append(energy)
        Time.append(time)

len_data=len(datas)
Enen=np.array(Ene).reshape(len_data,5,5).min(axis=2)
Timen=np.array(Time).reshape(len_data,5,5).mean(axis=2).mean(axis=1)

with open("2_neal_ene", "w") as file:
  for i in range(len_data):
    file.write('%s\n'%datas[i])
    for j in range(5):
      file.write('%12.5e ' % Enen[i][j]) 
    file.write('\n')
  file.write('\n')

with open("2_neal_time", "w") as file:
  for i in range(len_data):
    file.write('%s\n'%datas[i])
    file.write('%8.3f\n' % Timen[i]) 
  file.write('\n')


