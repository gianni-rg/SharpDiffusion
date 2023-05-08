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

// Based on 'Inference Stable Diffusion with C# and ONNX Runtime' sample code
// (https://github.com/cassiebreviu/StableDiffusion/)
// Copyright (C) 2023 Cassie Breviu.
// Licensed under the MIT License.

namespace SharpDiffusion.Schedulers;

using SharpDiffusion.Interfaces;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

public abstract class SchedulerBase : IScheduler
{
    protected readonly int _numTrainTimesteps;
    protected List<float> _alphasCumulativeProducts;
    public bool _isScaleInputCalled;

    public abstract List<int> Timesteps { get; protected set; }
    public abstract Tensor<float> Sigmas { get; protected set; }
    public abstract float InitNoiseSigma { get; protected set; }

    public SchedulerBase(int numTrainTimesteps = 1000)
    {
        _numTrainTimesteps = numTrainTimesteps;
        Timesteps = new List<int>();
    }

    protected static double[] Interpolate(double[] timesteps, double[] range, List<double> sigmas)
    {
        // Create an output array with the same shape as timesteps
        var result = np.zeros(timesteps.Length + 1);

        // Loop over each element of timesteps
        for (int i = 0; i < timesteps.Length; i++)
        {
            // Find the index of the first element in range that is greater than or equal to timesteps[i]
            int index = Array.BinarySearch(range, timesteps[i]);

            // If timesteps[i] is exactly equal to an element in range, use the corresponding value in sigma
            if (index >= 0)
            {
                result[i] = sigmas[index];
            }

            // If timesteps[i] is less than the first element in range, use the first value in sigmas
            else if (index == -1)
            {
                result[i] = sigmas[0];
            }

            // If timesteps[i] is greater than the last element in range, use the last value in sigmas
            else if (index == -range.Length - 1)
            {
                result[i] = sigmas[-1];
            }

            // Otherwise, interpolate linearly between two adjacent values in sigmas
            else
            {
                index = ~index; // bitwise complement of j gives the insertion point of x[i]
                double t = (timesteps[i] - range[index - 1]) / (range[index] - range[index - 1]); // fractional distance between two points
                result[i] = sigmas[index - 1] + t * (sigmas[index] - sigmas[index - 1]); // linear interpolation formula
            }

        }
        // Add 0.000 to the end of the result
        result = np.add(result, 0.000f);

        return result.ToArray<double>();
    }

    public Tensor<float> ScaleModelInput(Tensor<float> sample, int timestep)
    {
        // Get step index of timestep from TimeSteps
        int stepIndex = Timesteps.IndexOf(timestep);

        // Get sigma at stepIndex
        var sigma = Sigmas[stepIndex];
        sigma = (float)Math.Sqrt(Math.Pow(sigma, 2) + 1);

        // Divide sample tensor shape by sigma
        sample = TensorHelper.DivideTensorByFloat(sample.ToArray(), sigma, sample.Dimensions.ToArray());

        _isScaleInputCalled = true;

        return sample;
    }

    public abstract void SetTimesteps(int numInferenceSteps);

    public abstract DenseTensor<float> Step(Tensor<float> modelOutput, int timestep, Tensor<float> sample, int order = 4);
}