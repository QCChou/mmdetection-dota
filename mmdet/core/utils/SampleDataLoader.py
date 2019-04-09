import torch


class SampleParallelDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, concats=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.concats = concats

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch = []
        for datas in self.loader:
            batch.append(datas)
            if len(batch) == self.batch_size:
                concats = self.concats if self.concats is not None else [True for _ in range(len(batch[0]))]
                yield [torch.cat(inner[:-1], dim=0) if inner[-1] else inner[:-1] for inner in zip(*batch, concats)]
                batch = []

        if len(batch) > 0 and not self.drop_last:
            concats = self.concats if self.concats is not None else [True for _ in range(len(batch[0]))]
            yield [torch.cat(inner[:-1], dim=0) if inner[-1] else inner[:-1] for inner in zip(*batch, concats)]
