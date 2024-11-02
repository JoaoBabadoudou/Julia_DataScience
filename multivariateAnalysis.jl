## Multivariate analysis using iris dataset from read

using RDatasets
using StatsBase
using Plots


iris= dataset("datasets", "iris")
describe(iris)
unique(iris.Species)

## Linear regression

using DataFrames, GLM, Statistics, LinearAlgebra

xVals, yVals = iris[:,1], iris[:,3]
n = length(xVals)
A = [ones(n) xVals]

# Approach 
xBar, yBar = mean(xVals),mean(yVals)
sXX, sXY = ones(n)'*(xVals.-xBar).^2 , dot(xVals.-xBar,yVals.-yBar)
b1A = sXY/sXX
b0A = yBar - b1A*xBar
Y=(b1A*xVals) .+ b0A
X=xVals



scatter(xVals, yVals, msw=0)
xlims = [minimum(xVals), maximum(yVals)]
plot!(X, Y,
c=:black, legend=:none, xlabel = "Sepal Length (cm)", ylabel = "Petal Length (cm)")


