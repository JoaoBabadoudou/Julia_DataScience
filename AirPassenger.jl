############### The classic Box & Jenkins airline data. 
#########  Monthly totals of international airline passengers, 1949 to 1960

using Plots
using CSV
using TimeSeries
using DataFrames
data = CSV.read("D:/Julia_project/DataHandling/AirPassengers.csv", DataFrame)

rand(length(date))


AirPassengers=[ data[:,2]]

size(date)
size(AirPassengers)
Date(2001,10,1)

date= [ Date(1949,01,01): Date(1960,12,31) ]
Array


Airdata= TimeArray(date, AirPassengers)

Plots(AirPassengers)

function TimeArray(timestamp::Vector{D}, 
    values::AbstractArray{T,N},
    colnames::Vector{UTF8String}, 
    meta::Any)
    nrow, ncol = size(values, 1), size(values, 2)
    nrow != size(timestamp, 1) ? error("values must match length of 
    timestamp"):
    ncol != size(colnames,1) ? error("column names must match width of
    array"):
    timestamp != unique(timestamp) ? error("there are duplicate dates"): 
    ~(flipdim(timestamp, 1) == sort(timestamp) || timestamp ==
    sort(timestamp)) ? error("dates are mangled"): 
    flipdim(timestamp, 1) == sort(timestamp) ?
    new(flipdim(timestamp, 1), flipdim(values, 1), colnames, meta): 
    new(timestamp, values, colnames, meta)
    end 
   end