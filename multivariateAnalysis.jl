## Multivariate analysis using iris dataset from read

using RDatasets
using StatsBase
using Plots
using lin

iris= dataset("datasets", "iris")
describe(iris)
unique(iris.Species)

## Linear regression

using DataFrames, GLM, Statistics, LinearAlgebra, CSV

xVals, yVals = iris[:,1], iris[:,2]
n = length(xVals)
A = [ones(n) xVals]

# Approach 1
xBar, yBar = mean(xVals),mean(yVals)
sXX, sXY = ones(n)'*(xVals.-xBar).^2 , dot(xVals.-xBar,yVals.-yBar)
b1A = sXY/sXX
b0A = yBar - b1A*xBar
Y=(b1A*xVals) .+ b0A
X=xVals
plot(X, Y),
scatter(xVals,yVals)

# Approach 2
b1B = cor(xVals,yVals)*(std(yVals)/std(xVals))
b0B = yBar - b1B*xBar

