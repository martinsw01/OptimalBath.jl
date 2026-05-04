export MooncakeGradient, MooncakeBackend

import Mooncake

MooncakeBackend() = DI.AutoMooncake()
const MooncakeGradient{Preparation} = ADGradient{Preparation, <:DI.AutoMooncake}
MooncakeGradient(args...) = ADGradient(MooncakeBackend(), args...)