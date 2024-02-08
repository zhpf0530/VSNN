###
# module LoadMyData for loading data and  data standardization
###

module LoadMyData
export preprocess_data,load_data
using MAT
using Statistics

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
    function preprocess_data(x,y,xT,yT)
        y1mean=0;
        y1std=1;
        x1mean=mean(x,dims=2)
        x1std=std(x,dims=2)
        x1=(x .-x1mean)./x1std;
        y1mean=mean(y)
        y1std=std(y);
        y1=(y .-y1mean)./y1std;
        x2=(xT .-x1mean)./x1std;
        y2=(yT .-y1mean)./y1std;
        return x1,y1,x2,y2,y1mean,y1std
    end
    function load_data(str::String)
        corndata=matread(str)
        x=convert.(Float32,corndata["Xcal"]');
        y=convert.(Float32,corndata["ycal"]')
        xT=convert.(Float32,corndata["Xtest"]')
        yT=convert.(Float32,corndata["ytest"]')
        return x,y,xT,yT
    end

end
