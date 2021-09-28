#!/usr/bin/env python
# coding: utf-8

# # Introduction to JupyterHub
# *Written by Luke Chang & Jeremy Huckins*
# 
# In this course we will primarily be using python to learn about fMRI data analysis. All of the laboratories can be run on your own individual laptops once you have installed Python (preferably via an [anaconda distribution](https://www.anaconda.com/distribution/). However, the datasets are large and there can be annoying issues with different versions of packages and installing software across different operating systems. We will also occasionally be using additional software that will be called by Python (e.g., preprocessing). We have a docker container available that will contain all of the software and have created tutorials to [download the data](../content/Download_Data.ipynb). In addition, some of the analyses we will run can be very computationally expensive and may exceed the capabilities of your laptop. 
# 
# To meet these needs, Dartmouth's Research Computing has generously provided a dedicated server hosted on [Google Cloud Platform](https://cloud.google.com/products/?utm_source=google&utm_medium=cpc&utm_campaign=na-US-all-en-dr-bkws-all-all-trial-e-dr-1009892&utm_content=text-ad-none-any-DEV_c-CRE_491414382710-ADGP_Desk%20%7C%20BKWS%20-%20EXA%20%7C%20Txt%20~%20GCP%20~%20General_Technical_Product-KWID_43700060017821936-kwd-79074734358&utm_term=KW_gce%20google-ST_gce%20google&gclid=CjwKCAjw7rWKBhAtEiwAJ3CWLB-hCizREvT4mTSLhQWuz3rRccPmvOMFiZ0zKJs16AUi2EC-2920-BoC0OAQAvD_BwE&gclsrc=aw.ds) that will allow us to store data, access specialized software, and run analyses. This means that everyone should be able to run all of the tutorials on their laptops, tablets, etc by accessing notebooks on the jupyterhub server and will not need to install anything beyond a working browser.
# 
# ## Login
# The main portal to access this resource will be through the Jupyterhub interface. This allows you to remotely login in to the server through your browser at [https://jhub.dartmouth.edu](https://jhub.dartmouth.edu) using your netid. Note that you need to be either on campus or logged into VPN to access the portal. Please let us know if you are having difficulty logging in.
# 
# Once you've logged in you should see a screen like this.
# 
# ![](../images/jupyterhub/First_Time.png)
# 
# The `shared` folder contains all of the relevant notebooks and data for the course. Files in this folder are read only and cannot be modified by students. You will need to copy the file to a different folder on your server, which will cover below.
# 
# Every time you log in, jupyterhub will spin up a new server just for you and will update all of the files. Sometimes if a lot of new files have been added they may not have enough time to copy so you may need to log out and back in to make sure everything syncs correctly to your server.
# 
# ## Server
# Every student will be able to have their own personal server to work on. This server is running on Google cloud computing and should have all of the software you need to run the tutorials. If your server is idle for a short amount of time, it will automatically shut down. There are also a limited amount of resources available (e.g., storage, RAM). Each user has access to a small amount of RAM, keep an eye on how much your jobs are using. The server may crash if it exceeds a certain amount.

# ## Jupyter Notebooks
# 
# Jupyter notebooks are a great way to have your code, comments and results show up inline in a web browser. Work for this class will be done in Jupyter notebooks so you can reference what you have done, see the results and someone else could redo it in the future, similar to a typical lab notebook.
# 
# Rather than writing and re-writing an entire program, you can write lines of code and run them one at a time. Then, if you need to make a change, you can go back and make your edit and rerun the program again, all in the same window. In our specific case, we are going to use JupyterHub which lets several people access the same computer and data at the same time through a web browser. 
# 
# Finally, you can view examples and share your work with the world very easily through [nbviewer](https://nbviewer.jupyter.org).  One easy trick if you use a cloud storage service like dropbox is to paste a link to the dropbox file in nbviewer.  These links will persist as long as the file remains being shared via dropbox.
# 
# ***You will not be able to work directly and modify the notebooks in the `shared/PSYC60-21F/notebook` folder***. These will always be updating as we edit them. Instead, make sure you copy the notebooks you are working on to **your own personal folder**. These can only be viewed and changed by you and won't be deleted or updated when your server starts. We also recommend saving a backup of these files on your local computer, just in case anything happens to the cloud storage.

# ### Opening a notebook on the server
# 
# Click on `shared` ->  `PSYC60-21F` ->  `notebooks`. Click on any notebook you would like to open (e.g., `1_Introduction_to_Programming.ipynb`). It should open and you will be to view the notebook read-only, which means you will **NOT** be able to modify it.
# 

# ### Copying Notebook
# 
# For this course, you will need to create a copy of each notebook to your `homework` folder and work off of the copy. This will allow you to take risks and change the code as well as ensure that your work won't be deleted when the notebooks update. 
# 
# Let's learn how to duplicate the notebook and move it to the homework folder.
# 
# 1. In your home directory, you will need to create a folder called `homework`. Click on `New` then `Folder`.
# ![](../images/jupyterhub/New_Folder.png)
# 
# 2. You will need to rename the folder. `Check` the box next to the new untitled folder, then click `Rename`.
# ![](../images/jupyterhub/Rename_Folder.png)
# 
# Now type the new name of the folder, i.e., `homework`
# ![](../images/jupyterhub/Rename_Homework.png)
# 
# 3. Let's practice copying a notebook to your Homework folder. Navigate to `shared` -> `PSYC60-21F` -> `notebooks`, and open the notebook you would like to copy by clicking on it (e.g., `4_ICA.ipynb`. Then click `File` then `Save as...` button. 
# ![](../images/jupyterhub/SaveAs.png)
# 
# 4. You will need to specify the folder and name to save a copy of the notebook. Since we have already created the `homework` folder in your root directory of your server, we can now save notebooks to that location by specifying the path and the name of the file `homework/4_ICA`. 
# ![](../images/jupyterhub/SaveAs_Name.png)
# 
# Your duplicated notebook is now in your homework folder. You will need to do this for each new notebook assignment.
# 

# ## Alternative to Jupyterhub
# 
# If you use jupyter notebooks on your own computer then you own computer will be doing the processing. If you put your computer to sleep then processing will stop. It will also likely slow down other programs you are using on your computer. I would recommend installing it on your own computer so you can learn more about how to use it, or if you are interested in tinkering with the software or you happen to have a particularly fast/newer computer. We don't recommend going this route unless you don't have reliable access to the internet.
# 
# Please contact Professor Chang if you want any assistance doing this.
# 
# ### Installing Jupyter Notebooks on your own computer
# 
# 1. Install python. We recommend using the [Acaconda Distribution](https://www.anaconda.com/distribution/) as it comes with most of the relevant scientific computing packages we will be using.  Be sure to download Python 3.
# 
# Alternative 1: Install jupyter notebook (it comes with Anaconda)
# 
# ```
# pip install jupyter
# ```
# 
# Alternative 2: If you already have python installed:
# 
# ```
# pip install --upgrade pip
# ```
# 
# ```
# pip install jupyter
# ```
# 
# ### Starting Jupter Notebooks on your computer
# Open a terminal, navigate to the directory you want to work from then type `jupyter notebook` or `jupyter lab`
# 

# (python-packages)= 
# ### Python packages for the course
# All of the packages we will be using for the course are in the [requirements.txt](https://github.com/ljchang/dartbrains/blob/master/requirements.txt) file of the github repository.
# 
# You can install all of the packages using `pip`. We will download the file from github and parse it and install each package.

# In[24]:


import pandas as pd
import requests

requirements = requests.get('https://raw.githubusercontent.com/ljchang/dartbrains/master/requirements.txt').text
requirements = [x for x in requirements.split('\n')][3:-1]

for r in requirements:
    get_ipython().system('pip install {r}')


# ## Plotting and Atlases
# For most of our labs we will be using Python to plot our data and results.  However, it is often useful to have a more interactive experience.  We recommend additionally downloading [FSLeyes](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes), which is a standalone image viewer developed by FSL.  It can be installed by either downloading directly from the website, or using `pip`.
# 
# `pip install fsleyes` 
# 
# If you are using a mac, you will likely also need to add an X11 window system such as [xQuartz](https://www.xquartz.org/) for the viewer to work properly.

# ## References
# 
# [Jupyter Dashboard Walkthrough](https://365datascience.com/the-jupyter-dashboard-a-walkthrough/)
# 
# [Jupyter Notebook Manual](https://jupyter.readthedocs.io/en/latest/running.html#running)
# 
# [Getting Started With Jupyter Notebook](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46)
# 
# [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
# 
# [Convert jupyter notebook to slides](https://github.com/datitran/jupyter2slides)
