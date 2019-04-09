def accuracy_classification(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(target.shape) >= 2:
        _, target = target.topk(1, 1, True, True)

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    return res
