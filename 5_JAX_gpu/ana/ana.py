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
        elif "Steps" in line:
            step = int(re.search(r"[-+]?\d+", line).group())
    return energy, time, step

Ene=[]
Time=[]
Step=[]
datas=['1_data_1000','2_data_6000','3_data_25000']

for i in range (1,7,1):
  for data in datas:
    for j in range (1,6,1):
      for k in range (1,6,1):
        filename = f"../{i}_th_-{i}/{data}/1_solve/solution/{j}/jax_{k}"
        energy, time, step = read_pyqubo_results(filename)
        Ene.append(energy)
        Time.append(time)
        Step.append(step)

len_data=len(datas)
Enen=np.array(Ene).reshape(6,len_data,5,5).min(axis=3)
Timen=np.array(Time).reshape(6,len_data,5,5).mean(axis=3).mean(axis=2)
Stepn=np.array(Step).reshape(6,len_data,5,5).mean(axis=3).mean(axis=2)

with open("5_jax_ene", "w") as file:
  for i in range(len_data):
    file.write('%s\n'%datas[i])
    for m in range(6):
      for j in range(5):
        file.write('%12.5e ' % Enen[m][i][j]) 
      file.write('\n')
    file.write('\n')
  file.write('\n')

with open("5_jax_time", "w") as file:
  for i in range(len_data):
    file.write('%s\n'%datas[i])
    for m in range(6):
      file.write('%8.3f\n' % Timen[m][i]) 
    file.write('\n') 
  file.write('\n')

with open("5_jax_step", "w") as file:
  for i in range(len_data):
    file.write('%s\n'%datas[i])
    for m in range(6):
      file.write('%7d\n' % Stepn[m][i])
    file.write('\n')
  file.write('\n')

