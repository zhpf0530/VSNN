
module NN
    using Flux
    export flat_less,flat_less_conv,flat_less1,flat_less_conv1

    # ANN1
    function flat_less(input::Int,Dense1::Int)
        model = Chain(
            Dense(input => 25),          
            Dense(25 => Dense1),         
            Dense(Dense1 => 1))
        return model;
    end
    # ANN2
    function flat_less1(input::Int,Dense1::Int)
        model = Chain( 
            Dense(input => 25, elu),  
            Dense(25 => Dense1, elu),   
            Dense(Dense1 => 1))
        return model;
    end
    # CNN1
    function flat_less_conv(input::Int,Dense1::Int)
        model = Chain(
            Conv((5,),1=>1),
            Flux.flatten,
            Dense((input-4)*1=> 25),  
            Dense(25=> Dense1),  
            Dense(Dense1 => 1))
        return model;
    end
    # CNN2
    function flat_less_conv1(input::Int,Dense1::Int)
        model = Chain(
            Conv((5,),1=>1,elu),
            Flux.flatten,
            Dense((input-4)*1=> 25,elu),   
            Dense(25=> Dense1,elu),   
            Dense(Dense1 => 1))
        return model;
    end

end     