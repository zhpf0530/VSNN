# main.jl for test 
# suggest start julia with -t 6. example: julia -t 6

include("VSNN_M.jl")
include("LoadMyData.jl")
using .VSNN_M
using .LoadMyData
using DelimitedFiles
using JLD2

########################################################
# input
# fname: mat file name. eg. "corn_m5_moisture.mat"
# Nepoch: train loop umber of epoch
# wcoeff: weight coefficient(lamda2)
# resname: results file name(jld2)
# case1: 1-vector S 2-vector G 3-Vector Gw
# iter: iter th VSNN. 
# output  .jld2
# resname: file contains rmsec rmsecv rmsep and var_index selected
##########################################################
function var_select_NN(fname,Nepoch,wcoeff,resname,case1,iter)
    NModel=4;
    rmsecv=zeros(NModel)
    resep=zeros(NModel)
    rmsec=zeros(NModel)
    var_selected=[]
    for iModel=1:NModel
        t1=VSNN(fname,Nepoch,wcoeff,iModel,case1)
        inxdot=findlast('.',fname)
        save(fname[1:(inxdot-1)]*"Final"*"_"*string(iModel)*"_"*string(wcoeff)*"_"*string(case1)*"_"*string(iter,pad=3)*"model.jld2" ,"t1",t1)    
        (temp,tempi)=find_important_var(t1)
        rmsecv[iModel]=temp
        (temp,tempi2)=findmin(t1[1][tempi][1][1:end,1])
        rmsec[iModel]=t1[1][tempi][1][tempi2,1]
        resep[iModel]=t1[1][tempi][1][tempi2,2]
        push!(var_selected,t1[4][tempi]);
    end
    save(resname,Dict("rmsec"=>rmsec,"rmsecv"=>rmsecv,"resep"=>resep,"var_selected"=>var_selected))
end

# Calculate multiple repeated results
# N is iter count
function main(N)
    fname="corn_m5_moisture.mat"
    Nepoch=15000
    wcoeff=1e-5;
    for case1=1:3
        for iter=1:N
            resname="res_"*string(case1)*"_"*fname[1:9]*string(iter,pad=3)*".jld2"
            var_select_NN(fname,Nepoch,wcoeff,resname,case1,iter)
        end
    end
end

