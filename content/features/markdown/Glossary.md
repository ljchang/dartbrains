
# Glossary
Throughout this course we will use a variety of different functions available in the base Python library, but also many other libraries in the scientific computing stack. Here we provide a list of all of the functions that are used across the various notebooks. It can be a helpful reference when you are learning Python about the types of things you can do with various packages. Remember you can always view the docstrings for any function by adding a `?` to the end of the function name.

## Jupyter Cell Magic
%matplotlib inline
!pip

## Python Functions
import matplotlib.pyplot as plt
pwd
print
range
any()
int
str
glob.glob
os.path.join
os.path.basename
len
sort

## Pandas Functions
pd.read_csv()
pd.concat()
pd.DataFrame.isnull()
pd.DataFrame.mean()
pd.DataFrame.std()
pd.DataFrame.plot()
pd.DataFrame.map()
pd.DataFrame.groupby()
pd.DataFrame.fillna
pd.DataFrame.replace

## Matplotlib Functions
plt.figure
plt.subplots
plt.scatter
axis.set_xlim()
axis.set_ylim()
axis.set_xticklabels([str(int(tick)/1000)+'K' for tick in ax.get_xticks()])
axis.set_xlabel('Salary', fontsize=18)
axis.set_ylabel('Age', fontsize=18)
axis.set_title('Scatterplot of age and salary', fontsize=18)
f.savefig('MyFirstPlot.png')
plt.imshow(data.get_data()[:,:,50])
plt.tight_layout
plt.legend
bar
hist
axvline

## Seaborn Functions
sns.heatmap()
sns.catplot(x='departm', y='salary', hue='gender', data=df, ci=68, kind='bar')
sns.jointplot("age", "salary", data=df, kind='reg')
ax = sns.regplot(df.age, df.salary)

## nibabel Functions
nib.load
data.get_data()
data.shape
data.header
data.affine

## numpy Functions
np.array()
np.dot()
np.cos()
np.sin()
np.squeeze()
np.mean()
np.std()
np.random.randint()
np.random.randn()
np.zeros()
np.ones()
np.nan()
np.convolve()
np.pi()
np.arange
np.exp
np.real
np.fft.fft
np.fft.ifft
np.vstack
np.hstack
np.sqrt
np.diag
np.linalg.pinv
np.diag_indices

## networkx Functions
import networkx as nx
nx.draw_kamada_kawai(g)
nx.degree

## scipy Functions
scipy.signal.butter
scipy.signal.filtfilt
scipy.signal.freqz
scipy.signal.sosfreqz
scipy.stats.ttest_1samp
scipy.stats.binom

## nilearn Functions
nilearn.plotting.plot_anat
nilearn.plotting.view_img
nilearn.plotting.plot_glass_brain
nilearn.plotting.plot_stat_map

## nltools Functions
Brain_Data
fetch_pain
glover_hrf
Brain_Data.mean()
Brain_Data.std()
Brain_Data.shape()
Brain_Data.copy()
Brain_Data.to_nifti()
Brain_Data.append()
Brain_Data.write()
Brain_Data.plot()
Brain_Data.iplot()
ttest
extract_roi()
decompose
predict
apply_mask
data.find_spikes
stats['beta'].smooth(fwhm=6)
from nltools.stats import regress
regress
zscore
Design_Matrix
Adjacency
find_spikes
dm.convolve()
dm.heatmap()
dm.head()
dm.info()
dm.vif()
dm_conv.add_dct_basis
dm_conv_filt.add_poly()
SimulateGrid
fdr
threshold
from nltools.stats import fdr, one_sample_permutation
Adjacency.to_graph()
from nltools.stats import fdr, threshold, fisher_r_to_z, one_sample_permutation
Adjacency.plot_mds
distance
distance_to_similarity
similarity

## scikit learn Functions
from sklearn.metrics import pairwise_distances
pairwise_distances
balanced_accuracy_score method,

## jargon
tr
sampling frequency
hrf
Multicollinearity
glm
Orthogonalization
regression
convolution
filter
high-pass
low-pass
band-stop
band-pass
frequency
design matrix
estimator
residual
standard error
beta
hypothesis test
significance
contrasts
efficiency
inter-trial interval
jitter
autocorrelation
covariates
