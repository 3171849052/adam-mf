from dataclasses import dataclass
@dataclass(frozen=True)
class Method:
    label:str
    optimizer:str
    private:bool
    mechanism:str='none'
    workload:str='identity'
    scaled:bool=False
METHODS={
 'adam-iid':Method('Adam-IID','adam',True,'iid'),
 'momentum-iid':Method('Momentum-IID','momentum',True,'iid'),
 'adam-mf-sgd-b2lt':Method('Adam + SGD-B²LT','adam',True,'b2lt','sgd'),
 'adam-mf-sgd-stp-b2lt':Method('Adam + STP + SGD-B²LT','adam',True,'b2lt','sgd',True),
 'stpex-mf-momentum-bias-b2lt':Method('STP-EX + Momentum-Bias-B²LT','stpex',True,'b2lt','momentum-bias',True),
 'momentum-mf-momentum-b2lt':Method('Momentum + Momentum-B²LT','momentum',True,'b2lt','momentum'),
}
