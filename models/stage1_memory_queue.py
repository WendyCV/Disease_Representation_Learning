import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleMemoryQueue(nn.Module):
    def __init__(self, feature_dims, queue_size=4096):
        super().__init__()
        self.feature_dims = feature_dims
        self.queue_size = queue_size
        self.num_scales = len(feature_dims)

        self.register_buffer("ptr", torch.zeros(self.num_scales, dtype=torch.long))

        for i, dim in enumerate(feature_dims):
            q = torch.randn(dim, queue_size)
            q = F.normalize(q, dim=0)
            self.register_buffer(f"queue_{i}", q)

    def get_queue(self, i):
        return getattr(self, f"queue_{i}")

    @torch.no_grad()
    def enqueue_dequeue(self, keys_per_scale):
        """
        keys_per_scale: list of [B,D]
        """
        for i, keys in enumerate(keys_per_scale):
            keys = F.normalize(keys.detach(), dim=1)
            batch_size = keys.shape[0]

            queue = self.get_queue(i)
            ptr = int(self.ptr[i].item())

            if batch_size > self.queue_size:
                keys = keys[:self.queue_size]
                batch_size = self.queue_size

            if ptr + batch_size <= self.queue_size:
                queue[:, ptr:ptr + batch_size] = keys.T
            else:
                first = self.queue_size - ptr
                queue[:, ptr:] = keys[:first].T
                remain = batch_size - first
                queue[:, :remain] = keys[first:].T

            ptr = (ptr + batch_size) % self.queue_size
            self.ptr[i] = ptr

    def all_queues(self):
        return [self.get_queue(i) for i in range(self.num_scales)]