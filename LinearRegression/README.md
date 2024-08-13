# Linear Regression
## what is linear regression?
Linear Regression is a [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) algorithm in machine learning, which is widely used for solving regression problems. Regression is a type of machine learning problems where the goal is to predict a continuous output variable based on one or more input variables.


### example of linear regression usage
##### the popular example of linear regression usage is house price prediction

<img src="https://miro.medium.com/v2/resize:fit:1024/0*YMZOAO8QE4bZ4_Rk.jpg" width="500">


<h5>assume that there is a linear relationship between size of house and his price.</h5>
<h5>the goal of linear regression is to find the best fitting line to describe the relationship between the input (house size) and the output (house price).</h5>

<img src="img/linear_reg.png" width="500">

##### we have alotof lines. each line represented by m and c, m the slope defined by `m =  (y₂ - y₁) / (x₂ - x₁)` and c the intercept of the line with y axis
##### each line allows us to predict the price of each house, denoted by `x`, based on its size, denoted by `y`. by using the equation of the line `y = m*x + c`, but the predicted price may not be valid depends on the ability of this line to discribe the data, for example the orange line.
##### In the upper figure there are different lines and the best line is the red line. We choose this line by seeing the figure and comparing the lines, but the computer cannot do this, there are a set of complex operations that the computer performs to choose the best line.

<!-- <img src="img/linear_reg.png" width="500"> -->

<h5>we define a error function or cost function that count the average value of error for each line, the error is the difference between the real house price y_r and the predicted house price y_p</h5>

<!-- <h5>allmost of the time we can't get a line that holds all points of the data set.</h5> -->

##### `error = (y_p - y_r)^2`
##### `y_p = m*x + c`
##### `error = ((m*x+c)-y_r)^2`

##### We use the exponent `^2` to avoid a negative value because sometimes the difference `y_p - y_r` is positive and sometimes it is negative, 

<img src="https://miro.medium.com/v2/resize:fit:1400/1*jmd_lPcwkZ6QByMfv2itXg.png" width="500">

##### to count the average value of error we aplly this error function over all points of data, the exprission of average also named mean squared error  is shown in the image below

<img src="https://i.sstatic.net/MKVCl.png" width="500">

<h5>we compute the error in eash point and sume the errors. after that we divides the sume of errors over the number of points, finily we get the average value of error for our line</h5>

<!-- <h5>now we can compute the cost or error of any generated line the next steap is to find the line that givs the minimum error</h5> -->


