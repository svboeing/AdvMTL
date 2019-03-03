import tensorflow as tf
import os
hyper_losses = {}
for hyp in next(os.walk('./outputs/task_at_a_time'))[1]:
    dir = "./outputs/task_at_a_time/"+hyp+"/eval"
    try:
        file = os.listdir(dir)[0]
    except Exception:
        continue
    dir = dir+"/"+file
    print(dir)
    if dir == "./outputs/task_at_a_time/hyperparams_search_12_1e-05_0.5_0.001_424_424_/eval/events.out.tfevents.1551523247.vitim":
        continue
    losses = []
    for e in tf.train.summary_iterator(dir):
        for v in e.summary.value:

            if v.tag == 'eval_loss':
                #print("loss", v.simple_value),
                losses.append(v.simple_value)
            #if  v.tag == 'eval_accuracy':
            #    print("acc", v.simple_value)
    if len(losses) == 0:
        continue
    min_loss = min(losses)
    hyper_losses[hyp] = min_loss
#"./outputs/task_at_a_time/hyperparams_search_8_0.0001_0.01_0.001_768_424_/eval/events.out.tfevents.1551393346.vitim"

print(min(hyper_losses, key=hyper_losses.get))
print(hyper_losses["hyperparams_search_14_5.48386_0.5_0.001_424_424_"])
