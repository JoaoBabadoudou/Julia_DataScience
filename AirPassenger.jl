############### The classic Box & Jenkins airline data. 
#########  Monthly totals of international airline passengers, 1949 to 1960

using Plots
using CSV
using TimeSeries
using Dates
using StatsBase
using LinearAlgebra
using DataFrames, GLM, Statistics, CSV, Plots


data = CSV.read("D:/Julia_project/DataHandling/AirPassengers.csv", DataFrame)

AirPassengers=data[:,2]

date=  Date(1949):Month(1) :Date(1960,12)

Airdata= TimeArray(date,AirPassengers)
plot(Airdata,linecolor=[:black])

summarystats(AirPassengers)

# Model choosing

#Analytic method


Seas=reshape(AirPassengers,(12,12))
meann=map(mean,Seas[:,1:12])
vecMean=zeros(12)
eachindex(AirPassengers)
for i in 1:12
    
    vecMean[i] = mean(Seas[:,i])
    
end

vecMean

vecSd= zeros(12)
for i in 1:12
    vecSd[i]= sqrt(var(Seas[:,i]))
end
vecSd

using DataFrames
vlue=DataFrame(Mean=vecMean, Sd=vecSd)

yVals=vlue[:,1]
xVals=vlue[:,2]

lm1 = lm(@formula(Mean ~ Sd), vlue)
 
println("***Output of LM Model:")
println(lm1)


println("\n***Individual methods applied to model output:")
println("Deviance: ",deviance(lm1))
println("Standard error: ",stderror(lm1))
println("Degrees of freedom: ",dof_residual(lm1))
println("Covariance matrix: ",vcov(lm1))
pred(x) = coef(lm1)'*[1, x]
      SStotal = sum((yVals .- mean(yVals)).^2)

println("R squared (calculated in two ways):",r2(lm1),
               ",\t", 1 - deviance(lm1)/SStotal)

println("MSE (calculated in two ways: ",deviance(lm1)/dof_residual(lm1),
 ",\t",sum((pred.(vlue.Mean) - vlue.Sd).^2)/(size(vlue)[1] - 2))

xlims = [minimum(vlue.Sd), maximum(vlue.Sd)]
scatter(vlue.Sd, vlue.Mean, c=:blue, msw=0)
plot!(xlims, pred.(xlims),
                    c=:red, xlims=(xlims),
     legend=:none)

###### Graphic method

hv=zeros(12)
for i in 1:12
    hv[i]= maximum(Seas[:,i])
end

lv=zeros(12)
for i in 1:12
    lv[i]= minimum(Seas[:,i])
end

timed=Date(1949):Year(1) :Date(1960)
hightVal= TimeArray(timed, hv)
lowVal=TimeArray(timed,lv)

plot(Airdata,linecolor=[:black])
plot!(hightVal, linecolor=[:red] )
plot!(lowVal, linecolor=[:red] )


########## Month plot for seasonalities

years=[year(d) for d in date]
months=[month(d) for d in date]

df= DataFrame(yrars=years, months= months
 , value=AirPassengers) 

plot(xaxis= df[:,1],yaxis=df[:,3]  ,  group= years
 , label= months,  xlabel= "Month", ylabel="Value", 
 title="Seasonality Plot ", lw=2)############I'll come back




