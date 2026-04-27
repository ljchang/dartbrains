import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Resampling Statistics
    *Written by Luke Chang*

    Most the statistics you have learned in introductory statistics are based on parametric statistics and assume a normal distribution. However, in applied data analysis these assumptions are rarely met as we typically have small sample sizes from non-normal distributions. Though these concepts will seem a little foreign at first, I personally find them to be more intuitive than the classical statistical approaches, which are based on theoretical distributions. Our lab relies heavily on resampling statistics and they are amenable to most types of modeling applications such as fitting abstract computational models, multivariate predictive models, and hypothesis testing.

    There are 4 main types of resampling statistics:
     1. **bootstrap** allows us to calculate the precision of an estimator by resampling with replacement
     2. **permutation test** allows us to perform null-hypothesis testing by empirically computing the proportion of times a test statistic exceeds a permuted null distribution.
     3. **jackknife** allows us to estimate the bias and standard error of an estimator by creating samples that drop one or more samples.
     4. **cross-validation** provides a method to provide an unbiased estimate of the out-of-sample predictive accuracy of a model by dividing the data into separate training and test samples, where each data serves as both training and test for different models.

    In this tutorial, we will focus on the bootstrap and permutation test. Jackknifing and bootstrapping are both used to calculate the variability of an estimator and often provide numerically similar results. We tend to prefer the bootstrap procedure over the Jackknife, but there are specific use cases where you will want to use the jackknife. We will not be covering the jackknife in this tutorial, but encourage the interested reader to review the [wikipedia page](https://en.wikipedia.org/wiki/Resampling_(statistics)#Jackknife) for more information. We will also not be covering cross-validation as this is discussed in the multivariate prediction tutorial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bootstrap
    In statistics, we are typically trying to make inferences about the parameters of a population based on a limited number of randomly drawn samples. How reliable are the parameters estimated from this sample? Would we observe the same parameter if we ran the model on a different independent sample? [*Bootstrapping*](https://projecteuclid.org/download/pdf_1/euclid.aos/1176344552) offers a way to empirically estimate the precision of the estimated parameter by resampling with replacement from our sample distribution and estimating the parameters with each new subsample. This allows us to capitalize on naturally varying error within our sample to create a distribution of the range of parameters we might expect to observe from other independent samples. This procedure is reasonably robust to the presence of outliers as they should rarely be randomly selected across the different subsamples. Together, the subsamples create a distribution of the parameters we might expect to encounter from independent random draws from the population and allow us to assess the precision of a sample statistic. This technique assumes that the original samples are independent and are random samples representative of the population.

    Let's demonstrate how this works using a simulation.

    First, let's create population data by sampling from a normal distribution. For this simulation, we will assume that there are 10,000 participants in the population that are normally distributed $\mathcal{N}(\mu=50, \sigma=10)$.
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde


    def plot_distribution(traces, *, title=None, xlabel='Value', ylabel='Density', vlines=None, height=400):
        """Plot one or more distributions as histogram + KDE on a plotly figure.

        Args:
            traces: list of (data, label) tuples, or a single (data, label) for one trace.
            title: figure title.
            xlabel: x-axis label.
            ylabel: y-axis label.
            vlines: list of (x, color, dash, [label]) tuples for vertical reference lines.
                color defaults to 'black', dash defaults to 'solid'.
            height: figure height in pixels.
        """
        if isinstance(traces, tuple) and len(traces) == 2 and not isinstance(traces[0], tuple):
            traces = [traces]
        fig = go.Figure()
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (data, label) in enumerate(traces):
            data = np.asarray(data)
            color = palette[i % len(palette)]
            fig.add_trace(go.Histogram(
                x=data, histnorm='probability density', opacity=0.5,
                name=label, marker_color=color,
            ))
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            fig.add_trace(go.Scatter(
                x=x_range, y=kde(x_range), mode='lines',
                name=f'{label} KDE', line=dict(color=color, width=2),
                showlegend=False,
            ))
        if vlines:
            for vline in vlines:
                x = vline[0]
                color = vline[1] if len(vline) > 1 else 'black'
                dash = vline[2] if len(vline) > 2 else 'solid'
                label = vline[3] if len(vline) > 3 else None
                fig.add_vline(
                    x=x, line=dict(color=color, dash=dash, width=2),
                    annotation_text=label, annotation_position='top',
                )
        fig.update_layout(
            title=title, xaxis_title=xlabel, yaxis_title=ylabel,
            barmode='overlay', height=height, hovermode='x unified',
            margin=dict(l=60, r=20, t=50 if title else 20, b=50),
        )
        return fig


    _mean = 50
    _std = 10
    population_n = 10000
    population = _mean + np.random.randn(population_n) * _std
    print(f'Population Mean: {np.mean(population):.3}')
    print(f'Population Std: {np.std(population):.3}')
    plot_distribution((population, 'Population'), title='Population Distribution')
    return np, pd, plot_distribution, population


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's run a single experiment where we randomly sample 20 participants from the population. You can see that the mean and standard deviation of this distribution are fairly close to the population even though we are not full sampling the distribution.
    """)
    return


@app.cell
def _(np, plot_distribution, population):
    sample_n = 20
    sample = np.random.choice(population, size=sample_n, replace=False)
    print(f'Sample Mean: {np.mean(sample):.3}')
    print(f'Sample Std: {np.std(sample):.3}')
    plot_distribution(
        (sample, 'Single Sample'),
        title=f'Sample Distribution: n={sample_n}',
        vlines=[(np.mean(sample), 'blue', 'solid', 'mean')],
    )
    return sample, sample_n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's estimate the mean of this sample via bootstrapping 5,000 times to estimate our certainty in this estimate from our single small sample.
    """)
    return


@app.cell
def _(np, plot_distribution, sample, sample_n):
    _n_bootstrap = 5000
    bootstrap_means = []
    for _b in range(_n_bootstrap):
        bootstrap_means.append(np.mean(np.random.choice(sample, size=sample_n, replace=True)))
    bootstrap_means = np.array(bootstrap_means)
    print(f'Bootstrapped Mean: {np.mean(bootstrap_means):.3}')
    print(f'Bootstrapped Std: {np.std(bootstrap_means):.3}')
    plot_distribution(
        (bootstrap_means, 'Bootstrap'),
        title='Distribution of Bootstrapped Means',
        vlines=[(np.mean(bootstrap_means), 'blue', 'solid', 'mean')],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From this simulation, we can see that the mean of the bootstraps is the same as the original mean of the sample.

    How confident are we in the precision of our estimated mean? In other words, if we were to look through all 5,000 of our subsamples, how many of them would be close to 50.1? We can define a confidence interval to describe our uncertainty in our estimate. For example, we can use the percentile method to demonstrate the range of the estimated parameter in 95% of our samples. To do this we compute the upper and lower quantiles of our bootstrap estimates centered at 50% (i.e., 2.5% & 97.5%).
    """)
    return


@app.cell
def _(np, plot_distribution, sample, sample_n):
    _n_bootstrap = 5000
    bootstrap_means_1 = []
    for _b in range(_n_bootstrap):
        bootstrap_means_1.append(np.mean(np.random.choice(sample, size=sample_n, replace=True)))
    bootstrap_means_1 = np.array(bootstrap_means_1)
    _lower_bound = np.percentile(bootstrap_means_1, 2.5)
    _upper_bound = np.percentile(bootstrap_means_1, 97.5)
    print(f'Bootstrapped Mean: {np.mean(bootstrap_means_1):.3}')
    print(f'95% Confidence Intervals: [{_lower_bound:.3}, {_upper_bound:.3}]')
    plot_distribution(
        (bootstrap_means_1, 'Bootstrap'),
        title='Distribution of Bootstrapped Means',
        vlines=[
            (np.mean(bootstrap_means_1), 'blue', 'solid', 'mean'),
            (_lower_bound, 'red', 'dash', '2.5%'),
            (_upper_bound, 'red', 'dash', '97.5%'),
        ],
    )
    return (bootstrap_means_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The percentile method reveals that 95% of our bootstrap samples lie between the interval [47.4, 55.6]. While the percentile method is easy to compute and intuitive to understand, it has some issues. First, if the original sample was small and not representative of the population, the confidence interval may be biased and too narrow. Second, if the bootstrapped distribution is not symmetric and is skewed, the percentile based confidence intervals will not accurately reflect the distribution. Efron (1987) proposed the bias-corrected and accelerated bootstrap, which attempts to address these issues. We will not be explaining this in detail at the moment and encourage the interested reader to review the original [paper](https://www.jstor.org/stable/pdf/2289144.pdf).

    Now let's see how the bootstrap compares to if we had run real independent experiments. Let's simulate 1000 experiments where we randomly sample independent participants from the population and examine the distribution of the means from these independent experiments.
    """)
    return


@app.cell
def _(np, plot_distribution, population, sample_n):
    n_samples = 1000
    sample_means = []
    for _b in range(n_samples):
        sample_means.append(np.mean(np.random.choice(population, size=sample_n, replace=False)))
    sample_means = np.array(sample_means)
    _lower_bound = np.percentile(sample_means, 2.5)
    _upper_bound = np.percentile(sample_means, 97.5)
    print(f'Random Sample Mean: {np.mean(sample_means):.3}')
    print(f'95% Confidence Intervals: [{_lower_bound:.3}, {_upper_bound:.3}]')
    plot_distribution(
        (sample_means, 'Random Samples'),
        title='Distribution of Random Sample Means',
        vlines=[
            (np.mean(sample_means), 'blue', 'solid', 'mean'),
            (_lower_bound, 'red', 'dash', '2.5%'),
            (_upper_bound, 'red', 'dash', '97.5%'),
        ],
    )
    return (sample_means,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that the mean is closer to the population mean, but our certainty is approximately equal to what we estimated from bootstrapping a single sample.

    Finally, let's compare the bootstrapped distribution of 20 samples to the 1000 random samples.
    """)
    return


@app.cell
def _(bootstrap_means_1, plot_distribution, sample_means):
    plot_distribution(
        [(bootstrap_means_1, 'Bootstrap'), (sample_means, 'Random Samples')],
        title='Bootstrapped vs Randomly Sampled Precision Estimates',
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This technique is certainly not perfect, but it is very impressive how well we can estimate the precision of a population level statistic from a single small experiment using bootstrapping. Though our example focuses on estimating the mean of a population, this approach should work for many different types of estimators. Hopefully, you can see how this technique might be applied to your own work.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Permutation Test
    After we have estimated a parameter for a sample, we often want to perform a hypothesis test to assess if the observed distribution is statistically different from a null distribution at a specific alpha criterion (e.g., p < 0.05). This is called null hypothesis testing, and classical statistical tests, such as the t-test, F-test, and $\chi^2$ tests, rely on theoretical probability distributions. This can be problematic when your data are not well approximated by the theoretical distributions. Using resampling statistics, we can empirically evaluate the null hypothesis by randomly shuffling the labels and re-running the statistic. Assuming that the labels are exchangeable under the null hypothesis, then the resulting tests yield the exact significance levels, i.e., the number of times we observed our result by chance. This class of non-parametric tests are called permutation tests, and are also occasionally referred to as randomization, re-rerandomization, or exact tests. Assuming the data is exchangeable, permutation tests can provide a "p-value" for pretty much any test statistic regardless if the distribution is known. This provides a relatively straightforward statistic that is easy to compute and understand. Permutation tests can be computationally expensive and often require writing custom code. However, because they are independent they can be run in parallel using multiple CPUs or on a high performance computing system.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One Sample Permutation Test

    Let's simulate some data to demonstrate how to run a permutation test to evaluate if the mean of the simulated sample is statistically different from zero.

    We will sample 20 data points from a normal distribution, $\mathcal{N}(\mu=1, \sigma=1)$.
    """)
    return


@app.cell
def _(np, plot_distribution):
    _mean = 1
    _std = 1
    sample_n_1 = 20
    sample_3 = _mean + np.random.randn(sample_n_1) * _std
    print(f'Sample Mean: {np.mean(sample_3):.3}')
    print(f'Sample Std: {np.std(sample_3):.3}')
    plot_distribution(
        (sample_3, 'Sample'),
        title='Sample Distribution',
    )
    return sample_3, sample_n_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this example, the null hypothesis is that the sample does not significantly differ from zero. We can empirically evaluate this by randomly multiplying each sample by a $1$ or $-1$ and then calculating the mean for each permutation. This will yield an empirical distribution of null means and we can evaluate the number of times the mean of our sample exceeds the mean of the null distribution.
    """)
    return


@app.cell
def _(np, plot_distribution, sample_3, sample_n_1):
    n_permutations = 5000
    permute_means = []
    for _p in range(n_permutations):
        permute_means.append(np.mean(sample_3 * np.random.choice(np.array([1, -1]), sample_n_1)))
    permute_means = np.array(permute_means)
    _p_value = 1 - np.sum(permute_means < np.mean(sample_3)) / len(permute_means)
    print(f'Sample Mean: {np.mean(sample_3):.3}')
    print(f'Null Distribution Mean: {np.mean(permute_means):.3}')
    print(f'n permutations < Sample Mean = {np.sum(permute_means < np.mean(sample_3))}')
    print(f'p-value = {_p_value:.3}')
    plot_distribution(
        (permute_means, 'Null Distribution'),
        title='Permutation Null Distribution',
        vlines=[(np.mean(sample_3), 'red', 'dash', 'observed mean')],
    )
    return (n_permutations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see from a null distribution that it is very rare for us to randomly observe a mean of .953. In fact, this only occurred twice in 5,000 random permutations of the data, which makes our p-value = 0.0004. The precision of our p-value is tied to the number of permutations we run. More samples will allow us to observe a higher precision of our p-value, but will also increase our computation time. We tend to use 5,000 or 10,000 permutations as defaults.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two Sample Permutation Test
    When we were computed a one-sample permutation test above, we randomly multiplied each data point by a $1$ or $-1$ to create a null distribution. If we are interested in comparing two different groups using a permutation test, we can randomly swap group labels and can recompute the difference between the two distribution. This assumes the data are exchangeable.

    Let's start by simulating two different groups. Sample 1 is randomly drawn from this normal distribution, $\mathcal{N}(\mu=10, \sigma=5)$, while Sample 2 is drawn from this normal distribution $\mathcal{N}(\mu=7, \sigma=5)$.
    """)
    return


@app.cell
def _(np, plot_distribution):
    sample_n_2 = 50
    mean_1 = 10
    std_1 = 5
    mean_2 = 7
    std_2 = 5
    sample_1 = mean_1 + np.random.randn(sample_n_2) * std_1
    sample_2 = mean_2 + np.random.randn(sample_n_2) * std_2
    print(f'Sample1 Mean: {np.mean(sample_1):.3}')
    print(f'Sample1 Std: {np.std(sample_1):.3}')
    print(f'Sample2 Mean: {np.mean(sample_2):.3}')
    print(f'Sample2 Std: {np.std(sample_2):.3}')
    plot_distribution(
        [(sample_1, 'Sample 1'), (sample_2, 'Sample 2')],
        title='Sample Distributions',
    )
    return sample_1, sample_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok, now to compute a permutation test to assess if the two distributions are different, we need to generate a null distribution by permuting the group labels and recalculating the mean difference between the groups.
    """)
    return


@app.cell
def _(n_permutations, np, pd, plot_distribution, sample_1, sample_2):
    data = pd.concat([
        pd.DataFrame({'Group': np.ones(len(sample_1)), 'Values': sample_1}),
        pd.DataFrame({'Group': np.ones(len(sample_2)) * 2, 'Values': sample_2}),
    ], ignore_index=True)
    _values = data['Values'].values
    _groups = data['Group'].values
    permute_diffs = []
    for _p in range(n_permutations):
        _shuffled = np.random.permutation(_groups)
        permute_diffs.append(_values[_shuffled == 1].mean() - _values[_shuffled == 2].mean())
    permute_diffs = np.array(permute_diffs)
    difference = np.mean(sample_1) - np.mean(sample_2)
    _p_value = 1 - np.sum(permute_diffs < difference) / len(permute_diffs)
    print(f'Difference between Sample1 & Sample2 Means: {difference:.3}')
    print(f'n permutations < Sample Mean Difference = {np.sum(permute_diffs < difference)}')
    print(f'p-value = {_p_value:.3}')
    plot_distribution(
        (permute_diffs, 'Null Distribution'),
        title='Permutation Null Distribution',
        xlabel='Difference in Means',
        vlines=[(difference, 'red', 'dash', 'observed diff')],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The difference we observed between the two distributions does not occur very frequently by chance. By permuting the labels and recomputing the difference, we found that the sample difference exceeded the permutated label differences 4,996/5000 times.

    As long as your data are exchangeable, you can compute p-values for pretty much any type of test statistic using a permutation test. For example, to compute a p-value for a correlation between $X$ and $Y$ you would just shuffle one of the vectors and recompute the correlation. The p-value is the proportion of times that the correlation value exceeds the permuted correlations.

    We hope that you will consider using these resampling statistics in your own work. There are many packages in python that can help you calculate bootstrap and permutation tests, but you can see that you can always write your code fairly easily if you understand the core concepts.
    """)
    return


if __name__ == "__main__":
    app.run()
