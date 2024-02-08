###
# module VSNN_M for VSNN
# variable select for NN
###

module VSNN_M
include("./modelLib.jl")
include("./data.jl")
using .LoadDataMy
using Statistics
using Flux
using ProgressMeter
using .modleLib
using MLUtils
using GRUtils
using Random
using DelimitedFiles
using JLD2
using Dates
using Base.Threads
using Zygote

export VSNN,find_wcoeffB,calmodelB,cal_crosscvB,find_important_var,cal_variable_gradient,calRC,cal_variable_gradient2
    ################################################################## 
    # find lambda2 coefficients of neural networks
    # Input
    # fname: mat file name. eg. "corn_m5_moisture.mat"
    # Nepoch: train loop umber of epoch
    # mModel: mModel==1  flat_less;mModel==2  flat_less1;
    #         mModel==3  flat_less_conv;mModel==4  flat_less_conv1;
    # wcoeff: a list for wcoeff. eg . [1e-6;1e-5;1e-4;1e-3;1e-2;1e-1]
    # case1:  case1==1 vector S; case1==2 vector G; case1==3 vector Gw
    # Output
    # cv and a file
    ###################################################################
    function find_wcoeffB(fname,Nepoch,mModel,wcoeff,case1)
        x,y,xT,yT=load_data(fname)
        cv=zeros(length(wcoeff));
        for i=1:length(wcoeff)
            cv[i],~=cal_crosscvB(x,y,Nepoch,wcoeff[i],mModel,case1)
        end
        inxdot=findlast('.',fname)
        writedlm(fname[1:inxdot-1]*"Nepoch_"*string(Nepoch)*"mModel_"*string(mModel)*"cv.dat",cv)
        return cv
    end
    
    ##################################################################
    # VSNN Variable Selection Algorithm Based on Neural Network for Near-Infrared Spectral
    # Input
    # fname: mat file name. eg. "corn_m5_moisture.mat"
    # Nepoch: train loop umber of epoch
    # mModel: mModel==1  flat_less;mModel==2  flat_less1;
    #         mModel==3  flat_less_conv;mModel==4  flat_less_conv1;
    # wcoeff: a list for wcoeff. eg . [1e-6;1e-5;1e-4;1e-3;1e-2;1e-1]
    # case1:  case1==1 vector S; case1==2 vector G; case1==3 vector Gw
    # Output
    # res: model and result(a list- length(res)==Niter)
    # rcend: var important vector(a list-length(rcend)==Niter)
    # cv0: cv in each EDF(a list-length(cv0)==Niter)
    # resIndx: index of remaining variables in each EDF(a list-length(resIndx)==Niter)
    ###################################################################
    function VSNN(fname,Nepoch,wdcoff,mModel,case1)
        x,y,xT,yT=load_data(fname)
        res=[];
        rcend=[];
        rcmean=[];
        resIndx=[];
        cv0=[];
        numvar=size(x,1)
        allindex=collect(1:numvar)
        Niter=20;
        ratio=0.9;
        r0=1;
        r1=2/numvar;
        b=log(r0/r1)/(Niter-1);  
        a=r0*exp(b);
        for iop=1:Niter
            t1=calmodelB(x,y,xT,yT,Nepoch,wdcoff,mModel,false);
            cv,rcmean=cal_crosscvB(x,y,Nepoch,wdcoff,mModel,case1)
            push!(res,t1)
            push!(cv0,cv)
            push!(resIndx,allindex)
            push!(rcend,rcmean);
            ratio=a*exp(-b*(iop+1));   
            K=round(Int,numvar*ratio);  
            println(K)
            if mModel>2
                # if nModel==3, CNN is selected. CNN parameters 5,So K<5 
                if(K<5)
                    break;
                end
            end
            indxtemp=sortperm(rcmean[:],rev=true)
            indx=sort(indxtemp[1:K][:])
            x=x[indx,:]
            xT=xT[indx,:]
            allindex=allindex[indx,:]
            fig=plot(rcmean);
            display(fig);
            println("the $(iop)th iter ");
        end
        return res,rcend,cv0,resIndx;
    end

    ##################################################################
    # calmodelB: model train
    # Input
    # x: train samples
    # y:
    # xT: test samples 
    # yT: 
    # Nepoch: train loop umber of epoch
    # wcoeff: a list for wcoeff. eg . [1e-6;1e-5;1e-4;1e-3;1e-2;1e-1]
    # mModel: mModel==1  flat_less;mModel==2  flat_less1;
    #         mModel==3  flat_less_conv;mModel==4  flat_less_conv1;
    # iscrossCV: iscrossCV==true crosscv; iscrossCV=false train
    # Output
    # res: model and result(a list- length(res)==Niter)
    # rcend: var important vector(a list-length(rcend)==Niter)
    # cv0: cv in each EDF(a list-length(cv0)==Niter)
    # resIndx: index of remaining variables in each EDF(a list-length(resIndx)==Niter)
    ###################################################################
    function calmodelB(x,y,xT,yT,Nepoch,wdcoff,mModel,iscrossCV)
        x1,y1,x2,y2,ymean,stdy=preprocess_data(x,y,xT,yT)
        # batch_size initialise
        if size(x1,2)<=256
            lr=size(x1,2)/256*0.01
            batch_size=size(x1,2)
        else
            lr=0.01
            batch_size=256;
        end
        nnode=5;
        if mModel==1
            model=flat_less(size(x1,1),nnode);
            mbest=flat_less(size(x1,1),nnode);
        elseif mModel==2
            model=flat_less1(size(x1,1),nnode);
            mbest=flat_less1(size(x1,1),nnode);
        elseif mModel==3
            x1=reshape(x1,size(x1,1),1,size(x1,2)) 
            x2=reshape(x2,size(x2,1),1,size(x2,2)) 
            # model=flat_less_conv(size(x1,1),128)|>gpu;
            model=flat_less_conv(size(x1,1),nnode);
            mbest=flat_less_conv(size(x1,1),nnode);
        else
            x1=reshape(x1,size(x1,1),1,size(x1,2)) 
            x2=reshape(x2,size(x2,1),1,size(x2,2)) 
            # model=flat_less_conv1(size(x1,1),128)|>gpu; 
            model=flat_less_conv1(size(x1,1),nnode);
            mbest=flat_less_conv1(size(x1,1),nnode);
        end 

        loader = Flux.DataLoader((x1, y1), batchsize=batch_size, shuffle=true)  ;
        optim = Flux.setup(Flux.Optimiser(WeightDecay(wdcoff), Adam(lr)), model) 
        l1norm(x) = sum(abs, x)
        loss1(model,x,y)=mean(sum(abs2.(model(x).-y)))/length(y)  
        test1(model,x,y)=mean(sum(abs2.(model(x).-y)))/length(y)
        losses = zeros(Float32,Nepoch)
        testlosses=zeros(Float32,Nepoch)
        minlost=0;
        for epoch in 1:Nepoch
            for (x, y) in loader
                l, grads = Flux.withgradient(m -> loss1(m,x, y), model)
                Flux.update!(optim, model, grads[1])
            end
            testmode!(model)

            losses[epoch]=stdy.*sqrt(loss1(model,x1, y1))
            testlosses[epoch]=stdy.*sqrt(test1(model,x2,y2))
            if epoch==1
                model_state = Flux.state(cpu(model));
                Flux.loadmodel!(mbest, model_state);
                if iscrossCV==true
                    minlost=testlosses[epoch]
                else
                    minlost=losses[epoch]
                end
            end

            if iscrossCV==true
                if testlosses[epoch]<minlost
                    model_state = Flux.state(cpu(model));
                    Flux.loadmodel!(mbest, model_state);
                    minlost=testlosses[epoch]
                end

            else
                if losses[epoch]<minlost
                    model_state = Flux.state(model);
                    Flux.loadmodel!(mbest, model_state);
                    minlost=losses[epoch]
                end
            end

            trainmode!(model)
            if mod(epoch,500)==0
                println(epoch,",",losses[epoch],",",testlosses[epoch])
            end
        end
        testmode!(model)
        res=([losses testlosses],cpu(model),mbest)
    end

    ##################################################################
    # cal_crosscvB: Calculating variable importance vectors through cross validation
    # Input
    # x: train samples
    # y:
    # Nepoch: train loop umber of epoch
    # wcoeff: a list for wcoeff. eg . [1e-6;1e-5;1e-4;1e-3;1e-2;1e-1]
    # mModel: mModel==1  flat_less;mModel==2  flat_less1;
    #         mModel==3  flat_less_conv;mModel==4  flat_less_conv1;
    # case1:  case1==1 vector S; case1==2 vector G; case1==3 vector Gw
    # Output
    # res: cv
    # rctemp: variable importance vectors
    ###################################################################     
    function cal_crosscvB(x,y,Nepoch,wdcoff,mModel,case1)
        cv0=0;
        kfolds=5;
        rctemp=zeros(size(x,1))
        if Threads.nthreads()>1 # multiple Threads
            temp=[]
            for ((x_train, y_train), (x_val, y_val)) in MLUtils.kfolds((x,y), k=kfolds)
                r1=@spawn calmodelB(x_train,y_train,x_val,y_val,Nepoch,wdcoff,mModel,true)
                push!(temp,r1)
            end
            for k=1:length(temp)
                t1=fetch(temp[k])
                (~,tempi)=findmin(t1[1][1:end,2]) 
                cv0=cv0+t1[1][tempi,2]
                x1,y1,x2,y2,y1mean,y1std=preprocess_data(x,y,x,y)
                if case1==1
                    rc=cal_variable_importent_rd(t1[end],x1)
                    rc= rc .-y1;
                    rcmean=abs.(rc)   
                    rcmean=mean(rcmean,dims=2)    
                elseif case1==2
                    rc=cal_variable_gradient(t1[end],x1)
                    rcmean=abs.(rc)   
                    rcmean=mean(rcmean,dims=2)  
                elseif case1==3
                    rc=cal_variable_gradient(t1[end],x1)
                    rc=rc.*x1;
                    rcmean=abs.(rc)   
                    rcmean=mean(rcmean,dims=2)  
                end
                rcmean=rcmean.-minimum(rcmean)
                rcmean=rcmean./maximum(rcmean);
                rctemp=rctemp+rcmean;
            end
        else
            for ((x_train, y_train), (x_val, y_val)) in MLUtils.kfolds((x,y), k=5)
                t1=calmodelB(x_train,y_train,x_val,y_val,Nepoch,wdcoff,mModel,true)
                (~,tempi)=findmin(t1[1][1:end,2]) 
                cv0=cv0+t1[1][tempi,2]
                x1,y1,x2,y2,y1mean,y1std=preprocess_data(x,y,x,y)
                if case1==1
                    rc=cal_variable_importent_rd(t1[end],x1)
                    rc= rc .-y1;
                    rcmean=abs.(rc)   
                    rcmean=mean(rcmean,dims=2)    
                elseif case1==2
                    rc=cal_variable_gradient(t1[end],x1)
                    rcmean=abs.(rc)   
                    rcmean=mean(rcmean,dims=2)  
                elseif case1==3
                    rc=cal_variable_gradient(t1[end],x1)
                    rc=rc.*x1;
                    rcmean=abs.(rc)   
                    rcmean=mean(rcmean,dims=2)  
                endan=mean(rcmean,dims=2)  
                end

                rcmean=rcmean.-minimum(rcmean)
                rcmean=rcmean./maximum(rcmean);
                rctemp=rctemp+rcmean;
            end
        end
        rctemp=abs.(rctemp)
        rctemp=rctemp.-minimum(rctemp)
        rctemp=rctemp./maximum(rctemp);
        return cv0/kfolds,rctemp;
    end

       
    function find_important_var(t1)
        cv=t1[3]
        (temp,tempi)=findmin(cv)
        return (temp,tempi) 
    end


    # vector G variable_gradient_method1_f'
    function calRC(model,x1) 
        t1=size(x1)
        dz=1e-4;
        w=zeros(t1);
        for i=1:t1[2]
            for j=1:t1[1]
                xtemp=copy(x1[:,i]);
                xtemp[j]=x1[j,i]+dz;
                xtemp=reshape(xtemp,size(xtemp,1),1,size(xtemp,2))
                xtemp2=x1[:,i];
                xtemp2=reshape(xtemp2,size(xtemp2,1),1,size(xtemp2,2))
                w[j,i]=(model(xtemp)[1]-model(xtemp2)[1])/dz
            end
        end
        return w
    end

    # vector G variable_gradient_method2_f'
    function cal_variable_gradient(m1,x1)
        b1=zeros(size(x1));
        for i=1:size(x1,2)
            temp=gradient(x1[:,i]) do x
                x=reshape(x,size(x,1),1,1)
                m1(x)[1]
            end
            b1[:,i]=temp[1];
        end
        return  b1
    end

    #  variable_gradient_method3_f''
    function cal_variable_gradient2(m1,x1)
        b1=zeros(size(x1));
        for i=1:size(x1,2)
            temp=diaghessian(x1[:,i]) do x
                x=reshape(x,size(x,1),1,1)
                m1(x)[1]
            end
            b1[:,i]=temp[1];
        end
        return  b1
    end
    
    # vector S
    function cal_variable_importent_rd(m1,x1)
        b1=zeros(size(x1));
        for i=1:size(x1,2)
            x11=x1[:,i]
            for j=1:length(x11)
                x11[j]=0;
                x12=reshape(x11,size(x11,1),1,1)
                b1[j,i]=m1(x12)[1]
                x11[j]=x1[j,i];
            end
        end
        return  b1
    end
end
