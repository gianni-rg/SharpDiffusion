// Copyright (C) 2023 Gianni Rosa Gallina. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace SharpDiffusion.Interfaces;

using Microsoft.ML.OnnxRuntime.Tensors;

public interface IScheduler
{
    List<int> Timesteps { get; }
    float InitNoiseSigma { get; }
    void SetTimesteps(int numInferenceSteps);
    DenseTensor<float> Step(Tensor<float> modelOutput, int timestep, Tensor<float> sample, int order = 4);
    Tensor<float> ScaleModelInput(Tensor<float> sample, int timestep);
}