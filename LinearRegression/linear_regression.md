### Linear Regression
Linear Regression is a [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) algorithm in machine learning, which is widely used for solving regression problems. Regression is a type of machine learning problem where the goal is to predict a continuous output variable based on one or more input variables.

###### the popular example of linear regression is house price prediction
![](https://miro.medium.com/v2/resize:fit:1024/0*YMZOAO8QE4bZ4_Rk.jpg)

###### for example we find a linear relationship between size of house and his price. `x` represent the house size and `y` represent the house price. the goal of linear regression is to find the best fitting line to describe the relationship between the input (house size) and the output (house price). this line allow us to predict the `y` of every `x` by using line equation `y = m*x + c`  every line represented by m and c, m the slope defined by `m =  (y₂ - y₁) / (x₂ - x₁)` and c the intercept of the line with y axis

![](https://miro.medium.com/v2/resize:fit:1400/0*St4CVriw9ZsS3FJR.png)


###### allmost of the time we can't get a line that holds all points of the data set. for this we define a error function or cost function that count the average error value for this line, the error is the difference between the real house price and the predicted house price

![](https://miro.medium.com/v2/resize:fit:1400/1*jmd_lPcwkZ6QByMfv2itXg.png)