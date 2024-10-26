############### The classic Box & Jenkins airline data. 
#########  Monthly totals of international airline passengers, 1949 to 1960

using StatsBase,TimeSeries, Dates, Plots, Dates

using DataFrames, GLM, Statistics, CSV, LinearAlgebra



data = CSV.read("D:/Julia_project/Julia_Data/AirPassengers.csv", DataFrame)

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

for i in 1:12
    
    vecMean[i] = mean(Seas[:,i])
    
end

vecMean

vecSd= zeros(12)
for i in 1:12
    vecSd[i]= sqrt(var(Seas[:,i]))
end
vecSd


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
months1=[month(d) for d in date]

df= DataFrame(yrars=years, monthss= months1
 , value=AirPassengers) 
yrars=years

value=AirPassengers


m=["Jan","Feb", "Mar", "Apr", "May", "Jun",
"Jul","Aug","Sep", "Oct", "Nov", "Dec"]

months= repeat(m,12)


plot(yrars,value, c=[:yellow  :red :blue :green 
:brown :black :purple :darkgreen :orange :skyblue :gray :grey]
,group=months
 ,  xlabel= "Month", ylabel="value", 
 title="Seasonality Plot ", lw=3)


############ Holt winters forcasting

# Apply Holt-Winters forecasting with additive seasonality
hw_model = holt_winters(AirPassengers, season=:multiplicative, period=12)

# Forecast the next 12 periods
forecast = predict(hw_model, 12)


# Plot original time series
plot(time_series, label="Original Data", legend=:topright, title="Holt-Winters Forecast", xlabel="Time", ylabel="Values")

# Plot forecasted data
forecasted_series = [time_series; forecast]
plot!(forecasted_series, label="Forecast", linestyle=:dash)


# Customize the parameters if needed (alpha, beta, gamma)
hw_model_custom = holt_winters(time_series, season=:additive, period=12, alpha=0.3, beta=0.1, gamma=0.2)
forecast_custom = predict(hw_model_custom, 12)


### Step 7: Forecast Accuracy Evaluation
#You can evaluate the accuracy of your forecast
#using metrics like Mean Absolute Error (MAE) or
#Mean Squared Error (MSE).


# Assuming you have actual data for the forecasted period
actual = data.values[end-11:end]  # Replace with actual data for validation
mse = mean((forecast .- actual).^2)
println("Mean Squared Error: $mse")

#Using the `TSAnalysis.jl` package, Holt-Winters
#forecasting in Julia becomes straightforward for time
# series data with trend and seasonality. You can 
#adjust the seasonality type and
# smoothing parameters to suit your specific dataset.

using TSAnalysis
########### Sarima

acf_values = autocor(AirPassengers)  # 20 lags of autocorrelation

# Plot ACF
acf_plot = plot(acf_values, seriestype=:bar, label="ACF",
 legend=:topright, title="Autocorrelation Function",
  xlabel="Lags", ylabel="ACF")


# Seasonal ARIMA: SARIMA(p,d,q)(P,D,Q,m), m is seasonality period
model_sarima = fit(SARIMA, AirPassengers, (1, 1, 1)(0, 1, 1, 12))  # SARIMA with monthly seasonality
sarima_forecast = forecast(model_sarima, 12)

plot(data, label="Actual")
plot!(sarima_forecast, label="SARIMA Forecast")








