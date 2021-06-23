#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import cv2 as cv
import numpy as np
import onnxruntime


class MiDaSPredictor(object):
    def __init__(
        self,
        model_path='midas_predictor/midas_v2_1_small.onnx',
        model_type='small',
    ):
        if model_type == "large":
            self._net_w, self._net_h = 384, 384
        elif model_type == "small":
            self._net_w, self._net_h = 256, 256
        else:
            print(f"model_type '{model_type}' not implemented")
            assert False

        self._model = onnxruntime.InferenceSession(model_path)
        self._input_name = self._model.get_inputs()[0].name
        self._output_name = self._model.get_outputs()[0].name

    def __call__(
        self,
        image,
    ):
        x = copy.deepcopy(image)

        x = cv.resize(x, (self._net_h, self._net_w))
        x = x[:, :, [2, 1, 0]]  # BGR2RGB
        x = x.reshape(1, self._net_h, self._net_w, 3)
        x = x.astype('float32')
        x /= 255.0

        result = self._model.run([self._output_name], {self._input_name: x})[0]
        result = np.array(result).reshape(self._net_h, self._net_w)

        return result
