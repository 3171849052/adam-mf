import numpy as np
import torch
@torch.no_grad()
def evaluate(model,loader,device,task):
    model.eval();loss=correct=count=0
    for batch in loader:
        batch=tuple(t.to(device) for t in batch)
        inputs,targets=(batch[0],batch[1]) if task=='cv' else (batch[:3],batch[3])
        logits=model(inputs)
        loss+=float(torch.nn.functional.cross_entropy(logits,targets,reduction='sum'))
        correct+=int((logits.argmax(1)==targets).sum());count+=len(targets)
    if not np.isfinite(loss) or count==0:raise FloatingPointError('Invalid evaluation')
    return dict(accuracy=correct/count,loss=loss/count,validation_examples=count)
