

using Measures


using    DataFrames,    Clustering,    Gadfly,    Random 
Random.seed!(429)
mean_x2    =    mean(x2_mat,    dims=1) 
##    mean    center    the    cols
x2_mat_c    =    x2_mat    .-    mean_x2 
N    =    size(x2_mat_c)[1]
##    kmeans()    -    each    column    of    X    is    a    sample    -    requires    reshaping    x2 
x2_mat_t    =    reshape(x2_mat_c,    (2,N))
##    Create    data    for    elbow    plot 
k    =    2:8
df_elbow    =    DataFrame(k    =    Vector{Int64}(),    tot_cost    =    Vector{Float64}()) 
for    i    in    k
tmp    =    [i,    kmeans(x2_mat_t,    i;    maxiter=10,    init=:kmpp).totalcost    ] 
push!(df_elbow,    tmp)
end
##    create    elbow    plot
p_km_elbow    =    plot(df_elbow,    x    =    :k,    y    =    :tot_cost,    Geom.point,    Geom.line, 
Guide.xlabel("k"),    Guide.ylabel("Total    Within    Cluster    SS"),
Coord.Cartesian(xmin    =    1.95),    Guide.xticks(ticks    =    collect(2:8)))

##############################################################
using Clustering, RDatasets, Random, Plots; pyplot()
Random.seed!(0)

K=3
df = dataset("cluster", "xclara")
data = copy(convert(Array{Float64}, df)')

seeds = initseeds(:rand, data, K)
xclaraKmeans = kmeans(data, K, init = seeds)

println("Number of clusters: ", nclusters(xclaraKmeans))
println("Counts of clusters: ", counts(xclaraKmeans))

df.Group = assignments(xclaraKmeans)

p1 = scatter(df[:, :V1], df[:, :V2], c=:blue, msw=0)
  scatter!(df[seeds, :V1], df[seeds, :V2], markersize=12, c=:red, msw=0)

p2 = scatter( df[df.Group .== 1, :V1], df[df.Group .== 1, :V2], c=:blue, msw=0)
 scatter!( df[df.Group .== 2, :V1], df[df.Group .== 2, :V2], c=:red, msw=0)
scatter!( df[df.Group .== 3, :V1], df[df.Group .== 3, :V2], c=:green, msw=0)

plot(p1,p2,legend=:none,ratio=:equal,
    size=(800,400), xlabel="V1", ylabel="V2", margin = 5mm)


#################################

using RDatasets, Distributions, Random, LinearAlgebra
Random.seed!(0)

K=3
df = dataset("cluster", "xclara")
n,_ = size(df)
dataPoints = [convert(Array{Float64,1},df[i,:]) for i in 1:n]
shuffle!(dataPoints)

xMin,xMax = minimum(first.(dataPoints)),maximum(first.(dataPoints))
yMin,yMax = minimum(last.(dataPoints)),maximum(last.(dataPoints))

means = [[rand(Uniform(xMin,xMax)),rand(Uniform(yMin,yMax))] for _ in 1:K]
labels = rand(1:K,n)
prevMeans = -means

while norm(prevMeans - means) > 0.001
              prevMeans = means
          labels = [argmin([norm(means[i]-x) for i in 1:K]) for x in dataPoints]
          means = [sum(dataPoints[labels .== i])/sum(labels .==i) for i in 1:K]
 end

countResult = [sum(labels .== i) for i in 1:K]
println("Counts of clusters (manual implementation): ", countResult)