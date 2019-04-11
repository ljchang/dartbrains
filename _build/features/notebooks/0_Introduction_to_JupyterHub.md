---
redirect_from:
  - "/features/notebooks/0-introduction-to-jupyterhub"
interact_link: content/features/notebooks/0_Introduction_to_JupyterHub.ipynb
kernel_name: python3
title: 'Getting Started'
prev_page:
  url: /features/markdown/Labs
  title: 'Labs'
next_page:
  url: /features/notebooks/1_Introduction_to_Programming
  title: 'Introduction to Python'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Introduction to JupyterHub
*Written by Jeremy Huckins & Luke Chang*

In this course we will primarily be using python to learn about fMRI data analysis. Many of the laboratories can be run on your own individual laptops once you have installed Python (preferably via an [anaconda distribution](https://www.anaconda.com/distribution/). Often we will also be working with neuroimaging datasets, which can have large storage requirements. We will also occasionally be using additional software that will be called by Python (e.g., preprocessing). Finally, some of the analyses we will run can be very computationally expensive and may exceed the capabilities of your laptop.  To meet these needs, Dartmouth's Research Computing has generously provided a dedicated server on their high performance computing system that will allow us to store data, access specialized software, and run analyses. 

The main portal to access this resource will be through the Jupyterhub interface. This allows you to remotely login in to the server through your browser at https://jupyter.dartmouth.edu using your netid. Please let us know if you are having difficulty logging in as this might mean we need to have an account created for you.

A couple of quick notes. You will be sharing this server with your fellow classmates as well as a few other classes on campus. This server has a limited number of resources (currently 16 CPUs and 128gb of RAM). This means that we may not all be able to run our analyses at the same time or the server may slow down or possibly even crash.  To be a good user, please do not leave your notebook running when you aren't using it. This can be accomplished by using the *logout* button when you're done rather than simply closing your laptop or browser. 

# Today's Goals

1. Connect your laptop to your personal DartFS and Psych60 DartFS space

2. From a terminal window ssh into jupyter.dartmouth.edu

3. Create link to access Psych60 data from jupyterhub

4. Add modules so we can use MRI-specific software on the server

5. Create a folder for yourself in Psych60/students_output/your_netid 

6. Run a test notebook from your personal folder

# Data Storage via DartFS

We will host most of our data and tutorials for the class on DartFS. DartFS is a file storage system that can be accessed by most computers and devices provided that you are on a campus network (or logged in via VPN). You have a home directory and the instructions below are for accessing your home space. If you are enrolled in the class you *should* be authorized to access the Psych 60 space. Please let us know if that is not the case. You need to be connected to Dartmouth network to access DartFS. If you are off campus, use a VPN connection.

[Accessing DartFS on a mac](http://rc.dartmouth.edu/index.php/hrf_faq/how-to-access-dartfs-from-a-mac/)

[Accessing DartFS on a pc](http://rc.dartmouth.edu/index.php/hrf_faq/how-to-access-dartfs-from-a-pc/)

RC Linux systems:      /dartfs-hpc/rc/home/e/d12345e

RC Linux systems:      /dartfs/rc/lab/P/Psych60

Mac Finder path:  smb://dartfs-hpc.dartmouth.edu/rc/home/e/d12345e

Mac Finder path:  smb://dartfs.dartmouth.edu/rc/lab/P/Psych60

Windows UNC path:     \\dartfs-hpc.dartmouth.edu\rc\home\e\d12345e

Windows UNC path:     \\dartfs.dartmouth.edu\rc\lab\P\Psych60

Please see the email you received to determine the location of your home directory.

## Accessing Additional Software

To access addditional software beyond our class Python conda environment, we will need to set up a few things on the research computing servers.  This involves remotely logging in to the server using a terminal and adding a few software modules to your environment profile.

Using a terminal window, type the following commands in your terminal window:

```
ssh your_netid@jupyter.dartmouth.edu [where "your_netid" is your netid]
module initadd fsl
module initadd freesurfer
module initadd afni
module initadd ants
module initadd spm

ln –s  /dartfs/rc/lab/P/Psych60  Psych60

quota
ls .snapshot

cd Psych60/students_output 
mkdir your_netid  [where "your_netid" is your netid]
exit
```

# Introduction to Jupyter Notebooks

Jupyter notebooks are a great way to have your code, comments and results show up inline in a web browser. Work for this class will be done in Jupyter notebooks so you can reference what you have done, see the results and someone else could redo it in the future, similar to a typical lab notebook.

Rather than writing and re-writing an entire program, you can write lines of code and run them one at a time. Then, if you need to make a change, you can go back and make your edit and rerun the program again, all in the same window. In our specific case, we are going to use JupyterHub which lets several people access the same computer and data at the same time through a web browser. 

Finally, you can view examples and share your work with the world very easily through [nbviewer](https://nbviewer.jupyter.org).  One easy trick if you use a cloud storage service like dropbox is to paste a link to the dropbox file in nbviewer.  These links will persist as long as the file remains being shared via dropbox.

## Using Jupyter Notebooks on a server hosted by Dartmouth

If you use our server, all analyses you run will be performed on a server deep in the basement of a building on campus. This is great because you can start an analysis and let it run, checking in on the the results later by reconnecting to the server and opening the notebook you were running. The downside of this setup is that you will be sharing processing power with others, which may lead to some forced coffee breaks.

https://jupyter.dartmouth.edu

Log in with your netid and it should bring you to your home DartFS space.  Again, *don't forget to **logout** when you are done running your analysis*.

## Once you have logged in
You should see something like this:

![2019-04-11_07-59-08.png](attachment:2019-04-11_07-59-08.png)

You will likely not have much in there unless you have used your DartFS space before.



## Try Using the Terminal
Click on **New** and then **Terminal** to bring up a terminal window. It should look like this:

![2019-04-11_07-59-08.png](attachment:2019-04-11_07-59-08.png)

Type the commands into your terminal:
`pwd`, `ls`, `module list`

The result should look like this:

![2019-04-11_08-00-45.png](attachment:2019-04-11_08-00-45.png)

Next click the **Running** tab and and click on **Shutdown** next to terminal. This is how we close sessions that we are finished with.

## Connecting to Psych60 lab space from within Jupyterhub

You should see a Psych60 link under the **Files** tab. If you click on it you **should** be brought to the data and tutorial folder.

![2019-04-11_08-01-52.png](attachment:2019-04-11_08-01-52.png)

You will find a variety of tutorial notebooks in here along with data that we will use for our first few labs.

# Opening a notebook on the server

Click on Files, then the Psych60 link you created, then navigate to **nipype_tutorial/notebooks** and click on **introduction_jupyter-notebook.ipynb**

This will be the first notebook we will work with today. Save this into your folder in **students_output** before running anything (File, Save As)

Now work through this sample notebook. 

When you are finished, click on the **Running** tab in the main jupyter window and then the **Shutdown** button to shutdown the notebook. Close the notebook window and then click on the **logout** button.  


# Alternative

If you use jupyter notebooks on your own computer then you own computer will be doing the processing. If you put your computer to sleep then processing will stop. It will also likely slow down other programs you are using on your computer. I would recommend installing it on your own computer so you can learn more about how to use it, or if you are interested in tinkering with the software or you happen to have a particularly fast/newer computer.

Please contact Professor Huckins/Chang if you want any assistance doing this.

## Installing Jupyter Notebooks on your own computer

1. Install python (I recommend [Acaconda](https://www.continuum.io/downloads))

Alternative 1: Install jupyter notebook (it comes with Anaconda)

```
pip3 install jupyter
```

Alternative 2: If you already have python installed:

```
pip3 install --upgrade pip
```

```
pip3 install jupyter
```

### Starting Jupter Notebooks on your computer
Open a terminal, navigate to the directory you want to work from then type `jupyter notebook` or `jupyter lab`






# References

[Jupyter Dashboard Walkthrough](https://365datascience.com/the-jupyter-dashboard-a-walkthrough/)

[Jupyter Notebook Manual](https://jupyter.readthedocs.io/en/latest/running.html#running)

[Getting Started With Jupyter Notebook](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46)

[Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

[Convert jupyter notebook to slides](https://github.com/datitran/jupyter2slides)
