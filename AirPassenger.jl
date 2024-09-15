############### The classic Box & Jenkins airline data. 
#########  Monthly totals of international airline passengers, 1949 to 1960

using Plots
using CSV
using TimeSeries
using Dates
using StatsBase


data = CSV.read("D:/Julia_project/DataHandling/AirPassengers.csv", DataFrame)

AirPassengers=data[:,2]

date=  Date(1949,01,01):Month(1) :Date(1960,12,31)

Airdata= TimeArray(date,AirPassengers)
plot(Airdata,linecolor=[:black])

summarystats(AirPassengers)


