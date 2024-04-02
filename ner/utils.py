

def postprocess(predictions,labels,label_list):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()
    
    true_predictions=[
        [label_list[p] for (p,l) in zip(prediction,label) if l!=-100]
        for prediction,label in zip(predictions,labels)
    ]
    
    true_labels=[
        [label_list[l] for (p,l) in zip(prediction,label) if l!=-100]
        for prediction,label in zip(predictions,labels)
    ]
    
    return true_labels, true_predictions