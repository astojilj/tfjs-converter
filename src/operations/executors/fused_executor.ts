/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';
import {Activation} from '@tensorflow/tfjs-core/dist/ops/fused_ops';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {Node} from '../types';

import {OpExecutor} from './types';
import {getParamValue} from './utils';

export let executeOp: OpExecutor = (node: Node, tensorMap: NamedTensorsMap,
                                    context: ExecutionContext):
                                       tfc.Tensor[] => {
  switch (node.op) {
    case 'conv2dAddActivate': {
      const stride =
          getParamValue('strides', node, tensorMap, context) as number[];
      const pad = getParamValue('pad', node, tensorMap, context);
      const dataFormat =
          (getParamValue('dataFormat', node, tensorMap, context) as string)
              .toUpperCase();
      const dilations =
          getParamValue('dilations', node, tensorMap, context) as number[];
      const addParam =
          getParamValue('addParam', node, tensorMap, context) as tfc.Tensor;
      const activation =
          getParamValue('activation', node, tensorMap, context) as Activation;
      return [tfc.fused.conv2dAddActivate(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor3D |
              tfc.Tensor4D,
          getParamValue('filter', node, tensorMap, context) as tfc.Tensor4D,
          [stride[1], stride[2]], pad as 'valid' | 'same',
          dataFormat as 'NHWC' | 'NCHW', [dilations[0], dilations[1]], addParam,
          activation)];
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'fused';
