# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2019 Western Digital Corporation or its affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
import torchvision

from transforms import default_transform_fn


def collate_img_label_fn(sample):
    images = []
    labels = []
    lengths = []
    labels_with_tail = []
    max_num_obj = 0
    for image, label, length in sample:
        images.append(image)
        labels.append(label)
        lengths.append(length)
        max_num_obj = max(max_num_obj, length)
    for label in labels:
        num_obj = label.size(0)
        zero_tail = torch.zeros((max_num_obj - num_obj, label.size(1)), dtype=label.dtype, device=label.device)
        label_with_tail = torch.cat((label, zero_tail), dim=0)
        labels_with_tail.append(label_with_tail)
    image_tensor = torch.stack(images)
    label_tensor = torch.stack(labels_with_tail)
    length_tensor = torch.tensor(lengths)
    return image_tensor, label_tensor, length_tensor

def transformTargets(targets):
    bounding_box = targets["bounding_box"]
    max, ind = torch.max(bounding_box, dim=3)
    min, ind = torch.min(bounding_box, dim=3)
    c = (400 + ((max + min)/2) * 10) / 1.92
    w = (max - min) * 10 / 1.92
    target = torch.cat((c, w), dim=2)
    category = targets["category"]
    targets = torch.cat((target, category.unsqueeze(2).double()), dim=2).float()
    target_lengths = torch.tensor(targets.numpy().shape[:2])
    return targets, target_lengths


def transformImages(imgs):
    img_size = 416
    batch_tensor = []
    for batch in imgs:
        images = []
        for img_tensor in batch:
            images.append(img_tensor)
        image_tensor = torchvision.utils.make_grid(images, nrow=3, padding=0)
        transformed_tensor, label = default_transform_fn(img_size)(torchvision.transforms.ToPILImage()(image_tensor))
        batch_tensor.append(transformed_tensor)
    return torch.stack(batch_tensor)