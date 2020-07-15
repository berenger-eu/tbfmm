import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import pandas
import sys
import numpy
import random

def get_colors(name, colors, colorsdic):
    if name not in colorsdic:
        colorsdic[name] = colors[len(colorsdic)]
    return colorsdic[name]
    
if len(sys.argv) != 2:
    raise ValueError('You must provide the source csv.')

csv_file=sys.argv[1]
print("Will read ", csv_file)

res = pandas.read_csv(csv_file, index_col=-1)
res.head()
res.columns = res.columns.str.strip()

res = res.sort_values(['algoname', 'nbparticles', 'nbthreads'])

csv_file = csv_file.replace('.csv', '_csv')

##################################################################################

colors = [(0.1, 0.1, 0.1), (0.5, 0.5, 0.5), (0.035, 0.364, 0.764), (0.764, 0.035, 0.109), (0.423, 0.035, 0.764), (0.035, 0.764, 0.662), (0.874, 0.862, 0.364), (244./255., 66./255., 137./255.), (255./255., 11./255., 0./255.), (0./255., 222./255., 255./255.), (146./255., 218./255., 161./255.)]
colorsdic = {}
white = (1., 1., 1.)
markers = ['o', '^', 's', 'x', 'p', '*', 'd'] 

bar_width = 0.1
space_width = 0.25
figsize=(8, 4)
maxvaloffset=1

file_extension= ['.pdf', '.png']

#######################################################################
counter_data=0
label_data=list()
ticks=list()
maxduration=0.
global_config_idx=0

fig, ax = pyplot.subplots(figsize=figsize)

print(res)
res.reset_index(level=-1, inplace=True)
all_algoname = numpy.sort(res.algoname.unique())

for idx_algoname, algoname in enumerate(all_algoname):
    print("Algo name: " + str(algoname))
    
    all_nbparticles= numpy.sort(res[(res.algoname == algoname)].nbparticles.unique())
    
    for idx_nbparticles, nbparticles in enumerate(all_nbparticles):
        print("Nb particles: " + str(nbparticles))
        
        reference = res[(res.algoname == algoname) & (res.nbparticles == nbparticles) & (res.nbthreads == 1)].exectime.unique()
        print("reference: " + str(reference))
        
        all_nbthreads = res[(res.algoname == algoname) & (res.nbparticles == nbparticles)].nbthreads
        all_timings = res[(res.algoname == algoname) & (res.nbparticles == nbparticles)].exectime
        all_eff = reference/(all_timings*all_nbthreads)
        
        ax.plot(all_nbthreads, all_eff, marker=markers[idx_nbparticles], color=colors[idx_algoname], label=str(algoname).replace('TbfSmSpetabaruAlgorithm','SPETABARU').replace('TbfOpenmpAlgorithm','OpenMP 4.5') + ' - N = ' + str(nbparticles).replace('10000000','10M').replace('1000000','1M'))

ax.set_ylabel('Parallel efficiency')
ax.set_xlabel('Number of threads')
pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

for fex in file_extension:
    fig.savefig(csv_file + fex, bbox_inches='tight')
pyplot.clf()




