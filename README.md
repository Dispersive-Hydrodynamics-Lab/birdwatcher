# Birdwatcher

We're looking for Poisson distributed birds that take the form of solitons. I
use python 3.5 for all following analysis.

# Numerical Solution

We can examine the numerical solution of the conduit equation, and examine if a
bunch of uniformly distributed solitons will become a gas after enough time.

## Generating an Initial Profile

We first need to generate an initial profile to run through the solver. We can
do this by generating a large amount of uniformly distributed [gaussian
    functions](https://en.wikipedia.org/wiki/Gaussian_function) that are also
    uniformly distributed in height. Recall that a gaussian is defined as

$$
f(x) = a\exp\left(-\frac{{(x-b)}^2}{2c^2}\right)
$$

With code this is generated in the following way.

```python
def gaussian_approx(domain: np.ndarray, amplitude: float, position: float) -> np.ndarray:
    gaussian = amplitude * np.exp(-(domain - position) ** 2 / (4 * np.log(1 + amplitude)))
    return gaussian
```

Putting it together we can generate an entire initial profile in this way.

```python
domain = np.arange(0, 400, 0.001)
number_of_solitons = np.random.randint(25, 40)
soliton_heights = np.random.randint(1, 10, size=(number_of_solitons))
soliton_positions = np.random.choice(domain, size=number_of_solitons, replace=True)
gaussians = np.ones(len(domain))
derivatives = np.zeros(len(domain))
print('Generating {}'.format(number_of_solitons))
for i in range(number_of_solitons):
    gaussian = gaussian_approx(domain, soliton_heights[i], soliton_positions[i])
    derivative = get_derivative(domain, soliton_heights[i], soliton_positions[i])
    gaussians += gaussian
    derivatives += derivative
```

Which generates the following type of initial profile. (Not always this exact
profile, as it is generated at random.)

![png](./out.png)

## Experiment Setup

## Analyzing Experimental Data

Assuming that we now have a large amount of line fit experimental data stored in
`.mat` files, we can load it with python and perform analysis.

We can assume that the data has a phase portrait similar to this.

![png](./output/full_camera_data_phaseportrait.png)

### Identifying and Tracking Solitons

The final result looks like the following animation.

![png](./output/full_camera_data_solitons.gif)

Let's go over how we get to this result.

We have our raw data (as shown in the above phase portrait) where each column is
a snapshot of the conduit in time, and each row is a pixel from the line fitting
process (TODO: Discuss linefit and doublecheck row/column orientation). In the
above animation, you can note a faint spiky line underneath the solid blue line.
That is the raw data being plotted from our conduit at any given moment in time.

You'll also note that that line is very noisy, which is something we don't want.
To reduce the noise (and hopefully improve the accuracy of our detection
process) we use a one dimensional [gaussian
filter](http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm), which basically
uses the gaussian distribution (recall from above) and
[convolves](https://en.wikipedia.org/wiki/Convolution) with the raw noisy
signal. An intuitive way to think about this is to recall the concept of signal
cancellation, and understand that with a small gaussian we're simply cancelling
out this uniformly distributed noise. Once we do this, the result is the solid
blue line that we perform the analysis on.

The biggest drawback to this smoothing process is that it needs to be tuned. If
our smoothing kernel is too large, we'll lose all of our solitons, but if it's
too small, too much noise will remain. This will need a more sophisticated
solution to better identify a reasonable smoothing kernel.

Now that we have a smoothed dataset, we can find the peaks over each snapshot in
time. To do this, we use **something amazing**. We throw away the rightmost
(could be different based on orientation) peak which represents the incoming
data, as it will be a false positive. We also establish a base threshold and
throw away any peaks that are less than this threshold in height. TODO: explain
find peaks, explain pros and cons of this process, tuning parameters, etc.

At this point in the process we have a set of peak locations for each "frame" in
the experiment. We now need to keep track of where these peaks go, as they
represent solitons. If we can track each peak as it "moves" across our phase
space, we can keep an eye on its behavior. To track this, we simply look at each
peak for each frame, compare the locations to the next frame, and the closest
transitioned peak (using $L_2$ norm distance) is the next location of that
soliton. TODO: Talk about breakdown points

And now we're done, we have a way to track these solitons.

### Converting Tracked Solitons to Random Variables

Once we have an idea of where our solitons go, we can interpret the timing
between them crossing an arbitrary "line" somewhere at the top of the conduit as
a random variable. Theory tells us this will be $X\sim Poisson(\lambda)$. We can
empirically test this using the [Kolmogorov-Smirnov
test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) which
essentially finds the max distance between the predicted CDF and the empirical
CDF.

![png](./output/full_camera_data_cdf.png)

### Estimating $\lambda$

The issue with the above method (as you can probably imagine) is estimating our
$\lambda$, or the mean of the Poisson distribution. Currently we simply average
all timings and use that as the parameter, however something more sophisticated
may be required.

### Running the Analysis

```bash
┬─[william@fillory:~/Dropbox/Projects/birdwatcher]
╰─>$ ./birdwatcher.py -w -s conduit_edges.mat
140404430778152
Processing conduit_edges.mat
============================
Plotting Phase Portrait
Smoothing Data
Initializing Figure
Calculating Peaks: 100%|██████████████████████████████████| 24000/24000 [02:14<00:00, 178.62it/s]
Calculating Soliton Transitions: 100%|████████████████| 23999/23999 [00:00<00:00, 1144230.50it/s]
Animating
```
