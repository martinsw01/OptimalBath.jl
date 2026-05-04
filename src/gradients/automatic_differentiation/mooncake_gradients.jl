export MooncakeGradient, MooncakeBackend

import Mooncake

const MooncakeBackend() = DI.AutoMooncake()
const MooncakeGradient{Preparation} = ADGradient{Preparation, <:DI.AutoMooncake}
const MooncakeGradient(args...) = ADGradient(MooncakeBackend(), args...)