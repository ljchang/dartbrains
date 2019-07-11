---
redirect_from:
  - "/features/notebooks/9-glm"
interact_link: content/features/notebooks/9_GLM.ipynb
kernel_name: python3
title: 'Introduction to the General Linear Model'
prev_page:
  url: /features/notebooks/8_fmriprep_tutorial
  title: 'Introduction to Automated Preprocessing with fmriprep'
next_page:
  url: /features/notebooks/10_GLM_Single_Subject_Model
  title: 'Modeling Single Subject Data'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Introduction to the General Linear Model (GLM)

*Written by Luke Chang*

In this tutorial we will explore how we can use the general linear model (GLM) to make inferences about brain responses in a single subject.

We will explore the statistics in the context of a simple hypothetical experiment.

I encourage you to watch some short introductory videos on regression by Jeanette Mumford.
 - [Simple Linear Regression](https://www.youtube.com/watch?v=yLgPpmXVVbs)
 - [Matrix Algebra Basics](https://www.youtube.com/watch?v=fkZj8QoYjq8)
 - [Multiple Linear Regression](https://www.youtube.com/watch?v=qdOG7YMolmA)
 - [Hypothesis Testing](https://www.youtube.com/watch?v=ULeg3DH3g9w)
 - [Contrasts in Linear Models](https://www.youtube.com/watch?v=yLgPpmXVVbs&t=631s)
 - [Intepreting Regression Parameters](https://www.youtube.com/watch?v=uClfe4pLrCo)
 - [Mean Centering Regressors](https://www.youtube.com/watch?v=K4S576j90N8)
 
Also, videos from the Principles of fMRI Course by Tor Wager & Martin Lindquist.
 - [GLM](https://www.youtube.com/watch?v=GDkLQuV4he4&list=PLfXA4opIOVrGHncHRxI3Qa5GeCSudwmxM&index=18)
 - [GLM Estimation](https://www.youtube.com/watch?v=Ab-5AbJ8gAs&list=PLfXA4opIOVrGHncHRxI3Qa5GeCSudwmxM&index=23)

First, let's import all of the modules we will need for this tutorial.



{:.input_area}
```python
%matplotlib inline

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltools.stats import regress
from nltools.external import glover_hrf
```


# Simulate a voxel time course
To generate an intuition for how we use the GLM to make inferences in fMRI data analysis we will simulate a time series for a single voxel. A simulation means that we will be generating synthetic data that will resemble real data. However, because we know the ground truth of the signal, we can evaluate how well we can recover the true signal using a general linear model.

Imagine that we are interested in what region of the brain is involved in processing faces. We might show a variety of faces to a participant. Each presentation of a face will be a *trial*. Let's simulate what a design might look like with 5 trials of faces.

![faces](../../images/glm/faces.png)



{:.input_area}
```python
n_tr = 200
n_trial = 5
face = np.zeros(n_tr)
face[np.arange(10, n_tr, int(n_tr/n_trial))] = 1

def plot_timeseries(data, linewidth=3, labels=None):
    plt.figure(figsize=(20,5))
    plt.plot(data, linewidth=linewidth)
    plt.ylabel('Intensity', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.tight_layout()
    if labels is not None:
        plt.legend(labels, fontsize=18)
    
plot_timeseries(face)
```



{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_3_0.png)



We now have 5 events of a 2 second duration where a face is shown. If we scanned someone with this design, we might expect to see any region involved in processing faces, but how would we know which of these regions, if any, *selectively* process faces? Many of the regions we would observe are likely involved in processing *any* visual stimulus.

To rule out this potential confound, we would need at least one other condition that would serve as a visual control. Something that might have similar properties to a face, but isn't a face. 

One possibility is to create a visual stimulus that has all of the same visual properties in terms of luminance and color, but no longer resembles a face. Here is an example of the same faces that have been Fourier transformed, phase-scrambled, and inverse Fourier transformed. These pictures have essentially identical low level visual properties, but are clearly not faces.

![phase](../../images/glm/phase_scrambled.png)

However, one might argue that faces are a type of object and regions that are involved in higher visual processing such as object recognition might not be selective to processing faces. In this case, an additional visual control might be objects.

![objects](../../images/glm/objects.png)

So clearly we need at least one additional control condition.  Let's make it pictures of objects. 

To demonstrate that a region is processing faces and not simply lower level visual properties or objects more generally, we can search for regions that are selectively more activated in response to viewing faces relative to objects. This is called a *contrast* and is the basic principle of the subtraction method for controlling for potential experimental confounds. Because BOLD fMRI is a relative and not absolute measure of brain activity, the subtraction method is a key aspect of experimental design.

Figures are from Huettel, Song, & McCarthy (2008)



{:.input_area}
```python
n_tr = 200
n_trial = 5
face = np.zeros(n_tr)
face[np.arange(10, n_tr, int(n_tr/n_trial))] = 1
obj = np.zeros(n_tr)
obj[np.arange(20, n_tr, int(n_tr/n_trial))] = 1
voxel = np.vstack([face,obj]).T

plot_timeseries(voxel, labels=['Face', 'Object'])
```



{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_5_0.png)



Let's imagine that in a voxel processing face specific information we might expect to see a larger activation in response to faces. Maybe two times bigger?

In our simulation, these two values are parameters we are specifying to generate the data. Specifically they refer to the amplitude of the response to Faces and Houses within a particular region of the brain.



{:.input_area}
```python
n_tr = 200
n_trial = 5
face_intensity = 2
object_intensity = 1

face = np.zeros(n_tr)
face[np.arange(10, n_tr, int(n_tr/n_trial))] = face_intensity
obj = np.zeros(n_tr)
obj[np.arange(20, n_tr, int(n_tr/n_trial))] = object_intensity
voxel = np.vstack([face,obj]).T

plot_timeseries(voxel, labels=['Face', 'Object'])
```



{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_7_0.png)



Ok, now we have two conditions that are alternating over time. 

We know that the brain has a delayed hemodynamic response to events that has a particular shape, so we will need to convolve these events with an appropriate HRF function. Here, we will use the double-gamma HRF function.



{:.input_area}
```python
tr = 2
hrf = glover_hrf(tr, oversampling=1)
plt.plot(hrf, linewidth=3)
plt.ylabel('Intensity', fontsize=18)
plt.xlabel('Time', fontsize=18)
```





{:.output .output_data_text}
```
Text(0.5, 0, 'Time')
```




{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_9_1.png)



We will use `np.convolve` from numpy to perform the convolution.  The length of the convolved data will be the length of the time series plus the length of the kernel minus 1. To make sure everything is the same length, we will chop off the extra time off the convolved time series.



{:.input_area}
```python
face_conv = np.convolve(face, hrf, mode='same')
obj_conv = np.convolve(obj, hrf, mode='same')
voxel_conv = np.vstack([face_conv, obj_conv]).T

plot_timeseries(voxel_conv, labels=['Face', 'Object'])
```



{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_11_0.png)



While this might reflect the expected HRF response to a single event, real data is much noiser. It is easy to add different types of noise. For example, there might be a low frequency drift, autocorrelation, or possibly some aliased physiological artifacts.

For now, let's start with something simple, like independent white noise drawn from a random Gaussian distribution

$$\epsilon \sim \mathcal{N}(\mu,\,\sigma^{2})$$ 

where $\mu = 0$ and $\sigma = 0.15$



{:.input_area}
```python
sigma = 0.15
epsilon = sigma*np.random.randn(n_tr, 2)
voxel_conv_noise = voxel_conv + epsilon

plot_timeseries(voxel_conv_noise, labels=['Face', 'Object'])
```



{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_13_0.png)



Now this looks much more like real BOLD activity.

If we were to extract signal from a specific voxel we wouldn't know which condition was which, so let's combine these two signals into a single simulated voxel timeseries by adding the two vectors together with the `.sum()` method.



{:.input_area}
```python
Y = voxel_conv_noise.sum(axis=1)

plot_timeseries(Y)
```



{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_15_0.png)



# Construct Design Matrix

Now that we have our simulated voxel timeseries, let's try and see if we can recover the original signal using a general linear model in the form of:

$$Y = X\beta + \epsilon$$

where $Y$ is our observed voxel time series. $X$ is our model or design matrix, and is where we will specify a predicted response to each condition. $\beta$ is a vector of values that we will estimate to scale our model. $\epsilon$ is independent gaussian noise. This model is linear because we can decompose $Y$ into a set of features r regressors that are scaled by an estimated $\beta$ parameter and summed together. The $\epsilon$ parameter is not usually known and can also be estimated.

You may be wondering how our model is distinct from our simulated data. Remember when we simulated the data we specified 3 parameters - face amplitude, object amplitude, and $\epsilon$, we could have also added a mean, but for now, let's just assume that it is zero. When we fit our model to the simulated data, we will in theory be able to recover these three parameters. 



{:.input_area}
```python
n_tr = 200
n_trial = 5
face = np.zeros(n_tr)
face[np.arange(10, n_tr, int(n_tr/n_trial))] = 1
obj = np.zeros(n_tr)
obj[np.arange(20, n_tr, int(n_tr/n_trial))] = 1
intercept = np.ones(n_tr)
X = np.vstack([intercept, np.convolve(face, hrf, mode='same'), np.convolve(obj, hrf, mode='same')]).T

plot_timeseries(X)
```



{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_18_0.png)



We can write our model out so that it is very clear what we are doing.

$$Y = \beta_0 \cdot Intercept + \beta_1 \cdot Faces + \beta_2 \cdot Objects + \epsilon$$

We can also make a plot and rotate the timeseries, to better reflect the equation.

It should be clear how each of these components relate to the regression equation.



{:.input_area}
```python
f, a = plt.subplots(ncols=4, figsize=(6, 10), sharey=True)
a[0].plot(np.expand_dims(Y, axis=1), range(len(Y)))
a[1].plot(X[:,0], range(len(Y)))
a[2].plot(X[:,1], range(len(Y)))
a[3].plot(X[:,2], range(len(Y)))
a[0].set_ylabel('Time', fontsize=18)
a[0].set_title('Y', fontsize=20)
a[1].set_title('Intercept', fontsize=20)
a[2].set_title('Faces', fontsize=20)
a[3].set_title('Objects', fontsize=20)
plt.gca().invert_yaxis()
plt.tight_layout()
```



{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_20_0.png)



# Estimate GLM

There are several ways to estimate the parameters for our general linear model. The Ordinary Least Squares (OLS) estimator finds the $\hat\beta$ hyperplane that minimizes the error between the observed $Y$ and predicted $\hat Y$. 

This can be formulated using linear algebra as: 

$$\hat{\beta} = (X^T X)^{-1}X^TY$$

There is also maximum likelihood estimator, which is identical to the ordinary least squares estimator, when the error terms are normally distributed.

$$L(\beta, \sigma^2 | Y, X) = \displaystyle \prod_{i=1}^{n}\frac{1}{\sqrt(2\pi\sigma^2)} \cdot e^{-\frac{(Y_i - \beta X_i)^2}{2\sigma^2}}$$

where 

$$\mathcal{N}(0,\sigma^{2})$$

For this class, we will primarily be focusing on the Ordinary Least Squares Estimator. In fact, just to demonstrate that the math is actually relatively straightforward, we will write our own function for the estimator using the linear algebra formulation. In practice, we typically will use a premade function, which is usually slightly more computationally efficient and also will calculate standard errors, etc.



{:.input_area}
```python
def estimate_beta(X, Y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)

beta = estimate_beta(X, Y)

plt.bar(['Intercept','Faces', 'Objects'], beta)
plt.xlabel('Regressor', fontsize=18)
plt.ylabel('Beta Value', fontsize=18)

print('beta Faces - beta Objects: %.2f' % (beta[1]-beta[2]))
```


{:.output .output_stream}
```
beta Faces - beta Objects: 1.14

```


{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_22_1.png)



We can see that our model is working pretty well. We did not add a mean to the simulated timeseries, so our estimator correctly figures out that the intercept parameter should be zero. The model also correctly figured out that the scaling parameter for the faces regressor was 1 unit higher than the objects regressor.  However, these are sort of centered around the mean. 

Another way to evaluate how well our model is working is to plot our predicted $\hat Y$ on top of our simulated $Y$.

We can quantify the degree to which our model is accurately predicting the observed data by calculating the residual.

$$residual = Y - \hat Y$$



{:.input_area}
```python
predicted_y = np.dot(X, beta)

predicted_ts = np.vstack([Y, predicted_y]).T

plot_timeseries(predicted_ts, labels=['Simulated Voxel', 'Predicted Voxel'])

residual = Y - predicted_y

plot_timeseries(residual)
plt.title('Residual', fontsize=20)
```





{:.output .output_data_text}
```
Text(0.5, 1.0, 'Residual')
```




{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_24_1.png)




{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_24_2.png)



## Standard Errors

As you can see, we are doing a reasonable job recovering the original signals.

You may recall that we specified 3 parameters in our simulation 
 - a $\beta$ weight for faces
 - a $\beta$ weight for objects
 - an $\epsilon$ noise parameter.

The *standard error of the estimate* refers to the standard deviation of the residual.

Formally, this can be described as:

$$\hat \sigma = \sqrt{\frac{\displaystyle \sum_i^n(\hat Y_i - Y_i)^2}{n-k}}$$

where $n$ is the number of observations and $k$ is the total number of regressors. 

This number is essentially an estimate of the overall amount of error in the model or $\epsilon$. This error is assumed to be independent and normally distributed. The smaller the residual variance $\hat\sigma$ the better the fit of the model.

As you can see, the parameter is close, but slightly higher than the one we simulated.  This might be because we have relatively little data in our simulation.



{:.input_area}
```python
predicted_sigma = np.std(residual)

print("Standard Error of the Estimate: %.2f" % predicted_sigma)

plt.hist(residual)
```


{:.output .output_stream}
```
Standard Error of the Estimate: 0.22

```




{:.output .output_data_text}
```
(array([ 7., 13., 43., 47., 44., 29., 11.,  3.,  2.,  1.]),
 array([-0.52935402, -0.39206799, -0.25478196, -0.11749593,  0.0197901 ,
         0.15707613,  0.29436216,  0.43164819,  0.56893422,  0.70622026,
         0.84350629]),
 <a list of 10 Patch objects>)
```




{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_26_2.png)



## Explained Variance

Sometimes we want a single metric to quantify overall how well our model explained the data. There are many metrics that can provide a quantitative measure of *goodness of fit*.

Here we will calculate $R^2$ using the following formula:

$$R^2 = 1 - \frac{\displaystyle \sum_i^n(\hat y_i - y_i)^2}{\displaystyle \sum_i^n(y_i - \bar y)^2}$$

where $y_i$ is the measured value of the voxel at timepoint $i$, $\hat y_i$ is the predicted value for time point $i$, and $\bar y$ is the mean of the measured voxel timeseries.

$R^2$ will lie on the interval between $[0,1]$ and can be interpreted as percentage of the total variance in $Y$ explained by the model, $X$, where 1 is 100% and 0 is none.



{:.input_area}
```python
def r_square(Y, predicted_y):
    SS_total = np.sum((Y - np.mean(Y))**2)
    SS_residual = np.sum((Y - predicted_y)**2)
    return 1-(SS_residual/SS_total)

print("R^2: %.2f" % r_square(Y, predicted_y))
```


{:.output .output_stream}
```
R^2: 0.55

```

## Standard Error of $\beta$ estimates

We can also estimate the uncertainty of regression coefficients. The uncertainty of the beta parameters is quantified as a standard error around each specific estimate. 

$$\sigma = \sqrt{diag((X^TX)^{-1})} \cdot \hat \sigma$$

This is essentially a confidence interval around the $\beta_j$ estimate. One standard error, $1*\hat \sigma$ is approximately equivalent to a 68% confidence interval, while $2*\hat\sigma$ is approximately a 95% confidence interval.





{:.input_area}
```python
std_error = np.sqrt(np.diag((np.linalg.pinv(np.dot(X.T, X))))) * predicted_sigma

plt.bar(['Intercept','Faces', 'Objects'], beta, yerr = predicted_sigma)
plt.xlabel('Regressor', fontsize=18)
plt.ylabel('Beta Value', fontsize=18)

```





{:.output .output_data_text}
```
Text(0, 0.5, 'Beta Value')
```




{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_31_1.png)



## Statistical Significance

We could also perform a hypothesis test to evaluate if any of the regressors are statistically different from zero.

This exercise is simply meant to provide parallels to common statistical jargon. In practice, this is actually rarely done in neuroimaging analysis as we are typically more interested in making statistical inferences across the population rather than within a single participant.

The formula for calculating a t-statistic is very simple:

$$t = \frac{\hat \beta_j}{\hat \sigma_j}$$

where $\beta_j$ refers to the estimated parameter for a regressor $j$, and $\sigma_j$ refers to the standard error of regressor $j$.

$t$ values that are more than 2 standard errors away from zero are called *statistically significant*, which basically just means we are more confident that the estimate is stable and not just an artifact of small sample size. In general, we don't recommend reading too much into significance for individual $\beta$ estimates in single subject fMRI analysis.



{:.input_area}
```python
t = beta/std_error
t
```





{:.output .output_data_text}
```
array([-0.49497307, 14.50961077,  6.26452574])
```



## Contrasts

Contrasts are a very important concept in fMRI data analysis as they provide the statistical inference underlying the subtraction method of making inferences. See this [video](https://www.youtube.com/watch?v=yLgPpmXVVbs) by Jeanette Mumford or this [video](https://www.youtube.com/watch?v=7MibM1ATai4&list=PLfXA4opIOVrGHncHRxI3Qa5GeCSudwmxM&index=20) by Tor Wager & Martin Lindquist for a more in depth overview. 

Contrasts describe a linear combination of variables in a regression model whose coefficients add up to zero. This allows us to flexibly compare different experimental conditions.

For example, suppose we just wanted to know the magnitude of an effect for a single condition, such as the brain response to faces. We would create a contrast code that isolates the effect size (i.e., $\beta$ estimate for the face regressor)

If our GLM, was:

$$Y = \beta_0 \cdot Intercept + \beta_1 \cdot Faces + \beta_2 \cdot Objects$$

then, the corresponding contrast code or vector for faces would be:

[0, 1, 0]

The contrast code for the object condition would be:

[0, 0, 1]

and importantly the contrast *between* the face and object condition would be:

[0, 1, -1]

More simply, we are calculating the magnitude of the effect of the difference between viewing faces and objects in a single voxel.

To make this a little bit more clear, we will show a graphical representation of the design matrix to make it obvious what we are contrasting.




{:.input_area}
```python
sns.heatmap(X)

c1 = [0, 1, 0]
c2 = [0, 0, 1]
c3 = [0, 1, -1]

{(str(c),np.dot(beta, c)) for c in [c1, c2, c3]}
```





{:.output .output_data_text}
```
{('[0, 0, 1]', 0.8685208378484232),
 ('[0, 1, -1]', 1.143107787787324),
 ('[0, 1, 0]', 2.011628625635747)}
```




{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_35_1.png)



## Efficiency

We can estimate the efficiency, or the quality of an estimator for a specific experimental design or a hypothesis testing procedure. Efficiency is related to power, or the ability to detect an effect should one exist. However, unlike power, we can estimate efficiency from our design matrix and do not actually need to know the standard error for the model (unlike with power calculations). Specifically, efficiency is defined as the inverse of the sum of the estimator variances. For a more detailed explanation and general experimental design recommendations see this [overview](http://imaging.mrc-cbu.cam.ac.uk/imaging/DesignEfficiency) by Rik Henson, or this [blog post](https://theclevermachine.wordpress.com/tag/fmri-design-efficiency/) on efficiency in experimental designs.

$$e(c\hat\beta) = \frac{1}{c(X^TX)^{-1}c^T}$$

Reducing collinearity or covariance between regressors can increase design efficiency



{:.input_area}
```python
def contrast_efficiency(X, contrast):
    c = np.array(contrast)
    return 1/(np.dot(np.dot(c, np.linalg.pinv(np.dot(X.T, X))), c.T))

c1 = [0, 1, 0]
c2 = [0, 0, 1]
c3 = [0, 1, -1]

[contrast_efficiency(X, x) for x in [c1, c2, c3]]
```





{:.output .output_data_text}
```
[2.553683999619242, 2.5536839996192415, 1.3465769784519137]
```



## Varying the Inter-Trial Interval with Jittering





{:.input_area}
```python
# Set Simulation Parameters
n_tr = 200
n_trial = 5
face_intensity = 2
object_intensity = 1
sigma = 0.15

# Build Simulation
face_trials = np.random.randint(10, 180, n_trial)
obj_trials = np.random.randint(10, 180, n_trial)
face = np.zeros(n_tr)
face[face_trials] = 1
obj = np.zeros(n_tr)
obj[obj_trials] = 1
voxel_conv = np.vstack([np.convolve(face*face_intensity, hrf, mode='same'), np.convolve(obj*object_intensity, hrf, mode='same')]).T
epsilon = sigma*np.random.randn(n_tr, 2)
voxel_conv_noise = voxel_conv + epsilon

# Build Model
Y = voxel_conv_noise.sum(axis=1)
X = np.vstack([np.ones(len(face)), np.convolve(face, hrf, mode='same'), np.convolve(obj, hrf, mode='same')]).T

# Estimate Model
beta = estimate_beta(X, Y)
predicted_y = np.dot(X, beta)
predicted_sigma = np.std(residual)
predicted_ts = np.vstack([Y, predicted_y]).T
plot_timeseries(predicted_ts, labels=['Simulated Voxel', 'Predicted Voxel'])

# Estimate Contrast Efficiency
[contrast_efficiency(X, x) for x in [c1, c2, c3]]
```





{:.output .output_data_text}
```
[2.515579145518315, 2.5125914094384942, 1.449084912124725]
```




{:.output .output_png}
![png](../../images/features/notebooks/9_GLM_39_1.png)



## Autocorrelation

The BOLD signal has some intrinsic autocorrelation that varies with the length of the TR. Different software packages have provided varying solutions to this problem. For example, SPM implements an AR(1) model, which means that it trys to account for the fact that the signal is consistently correlated (i.e., autoregressive) with one lag. In practice, these will rarely change the beta estimates, but rather will adjust our standard errors around the estimates. As we will discuss soon, most group level analyses ignore these subject level, or first-level errors anyway. It is debatable if this is actually a good practice, but it reduces the importance of accounting for autocorrelation when looking at group level statistics in standard experimental design.

Another important thing to note is that there is some evidence that the AR(1) model can actually increase false positives, especiall in shorter TRs.  See this [paper](https://www.sciencedirect.com/science/article/pii/S1053811912003825) by Anders Ekland and colleagues for more details. Also, this is a helpful [blog post](https://mandymejia.com/2016/11/06/how-to-efficiently-prewhiten-fmri-timeseries-the-right-way/) discussing prewhitening. 

For the scope of this course we will largely be ignoring this issue, but I will plan to add some examples and simulations in the future.  For now, I encourage you to watch this video on [AR models](https://www.youtube.com/watch?v=Mb9LDzvhecY&list=PLfXA4opIOVrGHncHRxI3Qa5GeCSudwmxM&index=24)

## Additional Resources for Learning about Regression

- To go much more in-depth on linear regression, read Chapter 3 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/). Alternatively, watch the [related videos](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/) or read a [quick reference guide](http://www.dataschool.io/applying-and-interpreting-linear-regression/) to the key points in that chapter.
- This [introduction to linear regression](http://people.duke.edu/~rnau/regintro.htm) is much more detailed and mathematically thorough, and includes lots of good advice.
- This is a relatively quick post on the [assumptions of linear regression](http://pareonline.net/getvn.asp?n=2&v=8).

# Exercises

For our homework exercises, let's use the simulation to answer questions we might have about experimental design.

## 1. What happens when we vary the signal amplitude?

Some signals will be very strong and others weaker. How does the contrast efficiency change when the signal amplitudes are stronger and weaker?

In this exercise, make a plot showing how the contrast efficiency changes over 3 different levels of signal intensity.

## 2. What happens when we vary the noise?

How does the amount of noise in the data impact our contrast efficiency?  

In this exercise, make a plot showing the contrast efficiency for 3 different levels of simulated noise.

## 3. How many trials do we need?

A common question in experimental design is determining the optimal number of trials.

In this exercise, try evaluating how 3 different numbers of trials might impact the contrast efficiency.

## 4. What is the impact of the stimulus duration?

What if one condition simply results in processes that systematically take longer than the other condition?

In this exercise, let's try to answer this question by creating a simulation where the signal intensity between the two condition is identical, but one simply has a longer duration (i.e., the duration has more TRs than the other condition). 

Make a plot showing what happens to the $\beta$ estimates.
