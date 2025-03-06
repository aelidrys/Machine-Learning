<h1>ft_linear_regression</h1>

<h3>ft_linear_regression is one of the project of 42-network here is his <a href='https://cdn.intra.42.fr/pdf/pdf/102546/en.subject.pdf'>Subject</a></h3>
<h3>The aim of this project is to introduce you to the basic concept behind machine learning.
For this project, you will have to create a program that predicts the price of a car by
using a linear function train with a gradient descent algorithm.</h3>

---

<h2>Code Steps</h2>

- <h4>first step is spliting the data to train set and test set the object of behind this action is train with the training set and keep test set for testing the final model.</h4>

  - <h5>80% of data for training and keep 20% for testing</h5>
  - <h5>fix random state in 88 wich give good performence (r5ndom state control how data will shufled)</h5>

- <h4>second step Scaling data by using MinMaxScaler method that trnsform data to a range between 0 and 1 this help learning rate
  to control the jumps of gradient descent also avoid large numbers</h4>

- <h4>Trianing with <a href=''>Gradient Descent</a> algorithm</h4>

- <h4>Visualizing to see if our model undestand the data or not</h4>

  - <h5>use matplotlib.pyplot library to show the actual points and our model prediction</h5>

- <h4>Save the wights in a file.csv to predict the price of a car for a new given mileage</h4>

