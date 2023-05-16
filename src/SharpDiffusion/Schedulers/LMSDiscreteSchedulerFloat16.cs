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

using MathNet.Numerics;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

public class LMSDiscreteSchedulerFloat16 : SchedulerBase<Float16>
{
    private readonly string _predictionType;

    public List<Tensor<Float16>> Derivatives;

    public override List<int> Timesteps { get; protected set; }
    public override Tensor<Float16> Sigmas { get; protected set; }
    public override float InitNoiseSigma { get; protected set; }

    public LMSDiscreteSchedulerFloat16(int numTrainTimesteps = 1000, float betaStart = 0.00085f, float betaEnd = 0.012f, string betaSchedule = "scaled_linear", string predictionType = "epsilon", List<float>? trainedBetas = null) : base(numTrainTimesteps)
    {
        _predictionType = predictionType;

        Derivatives = new List<Tensor<Float16>>();


        var alphas = new List<float>();
        var betas = new List<float>();

        if (trainedBetas != null)
        {
            betas = trainedBetas;
        }
        else if (betaSchedule == "linear")
        {
            betas = Enumerable.Range(0, numTrainTimesteps).Select(i => betaStart + (betaEnd - betaStart) * i / (numTrainTimesteps - 1)).ToList();
        }
        else if (betaSchedule == "scaled_linear")
        {
            var start = (float)Math.Sqrt(betaStart);
            var end = (float)Math.Sqrt(betaEnd);
            betas = np.linspace(start, end, numTrainTimesteps).ToArray<float>().Select(x => x * x).ToList();

        }
        else
        {
            throw new Exception("betaSchedule must be one of 'linear' or 'scaled_linear'");
        }

        alphas = betas.Select(beta => 1 - beta).ToList();

        _alphasCumulativeProducts = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b)).ToList();

        // Create sigmas as a list and reverse it
        var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();

        // Standard deviation of the initial noise distribution
        InitNoiseSigma = (float)sigmas.Max();
    }

    public override void Dispose()
    {
    }

    private double GetLmsCoefficient(int order, int t, int currentOrder)
    {
        // Compute a linear multistep coefficient

        double LmsDerivative(double tau)
        {
            double prod = 1.0;
            for (int k = 0; k < order; k++)
            {
                if (currentOrder == k)
                {
                    continue;
                }
                prod *= (tau - Sigmas[t - k]) / (Sigmas[t - currentOrder] - Sigmas[t - k]);
            }
            return prod;
        }

        double integratedCoeff = Integrate.OnClosedInterval(LmsDerivative, Sigmas[t], Sigmas[t + 1], 1e-4);

        return integratedCoeff;
    }

    public override void SetTimesteps(int numInferenceSteps)
    {
        double start = 0;
        double stop = _numTrainTimesteps - 1;
        double[] timesteps = np.linspace(start, stop, numInferenceSteps).ToArray<double>();

        Timesteps = timesteps.Select(x => (int)x).Reverse().ToList();

        var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
        var range = np.arange(0, (double)sigmas.Count).ToArray<double>();
        sigmas = Interpolate(timesteps, range, sigmas).ToList();
        Sigmas = new DenseTensor<Float16>(sigmas.Count);
        for (int i = 0; i < sigmas.Count; i++)
        {
            Sigmas[i] = (Float16)sigmas[i];
        }
    }

    public override DenseTensor<Float16> Step(Tensor<Float16> modelOutput, int timestep, Tensor<Float16> sample, int order = 4)
    {
        int stepIndex = Timesteps.IndexOf(timestep);
        var sigma = Sigmas[stepIndex];

        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        Tensor<Float16> predOriginalSample;

        // Create array of type float length modelOutput.length
        var predOriginalSampleArray = new Float16[modelOutput.Length];
        var modelOutPutArray = modelOutput.ToArray();
        var sampleArray = sample.ToArray();

        if (_predictionType == "epsilon")
        {
            for (int i = 0; i < modelOutPutArray.Length; i++)
            {
                predOriginalSampleArray[i] = sampleArray[i].Subtract(sigma.Mul(modelOutPutArray[i]));
            }
            predOriginalSample = TensorHelper.CreateTensor(predOriginalSampleArray, modelOutput.Dimensions.ToArray());
        }
        else if (_predictionType == "v_prediction")
        {
            //predOriginalSample = modelOutput * ((-sigma / Math.Sqrt((Math.Pow(sigma,2) + 1))) + (sample / (Math.Pow(sigma,2) + 1)));
            throw new Exception($"predictionType given as {_predictionType} not implemented yet.");
        }
        else
        {
            throw new Exception($"predictionType given as {_predictionType} must be one of `epsilon`, or `v_prediction`");
        }

        // 2. Convert to an ODE derivative
        var derivativeItems = new DenseTensor<Float16>(sample.Dimensions.ToArray());

        var derivativeItemsArray = new Float16[derivativeItems.Length];

        for (int i = 0; i < modelOutPutArray.Length; i++)
        {
            //predOriginalSample = (sample - predOriginalSample) / sigma;
            derivativeItemsArray[i] = sampleArray[i].Subtract(predOriginalSampleArray[i]).Div(sigma);
        }
        derivativeItems = TensorHelper.CreateTensor(derivativeItemsArray, derivativeItems.Dimensions.ToArray());

        Derivatives?.Add(derivativeItems);

        if (Derivatives?.Count > order)
        {
            // remove first element
            Derivatives?.RemoveAt(0);
        }

        // 3. compute linear multistep coefficients
        order = Math.Min(stepIndex + 1, order);
        var lmsCoeffs = Enumerable.Range(0, order).Select(currOrder => GetLmsCoefficient(order, stepIndex, currOrder)).ToArray();

        // 4. compute previous sample based on the derivative path
        // Reverse list of tensors this.derivatives
        var revDerivatives = Enumerable.Reverse(Derivatives).ToList();

        // Create list of tuples from the lmsCoeffs and reversed derivatives
        var lmsCoeffsAndDerivatives = lmsCoeffs.Zip(revDerivatives, (lmsCoeff, derivative) => (lmsCoeff, derivative));

        // Create tensor for product of lmscoeffs and derivatives
        var lmsDerProduct = new Tensor<Float16>[Derivatives.Count];

        for (int m = 0; m < lmsCoeffsAndDerivatives.Count(); m++)
        {
            var (lmsCoeff, derivative) = lmsCoeffsAndDerivatives.ElementAt(m);
            // Multiply to coeff by each derivatives to create the new tensors
            lmsDerProduct[m] = TensorHelper.MultipleTensorByFloat(derivative.ToArray(), (Float16)lmsCoeff, derivative.Dimensions.ToArray());
        }

        // Sum the tensors
        var sumTensor = TensorHelper.SumTensors(lmsDerProduct, new[] { modelOutput.Dimensions[0], 4, 64, 64 });

        // Add the summed tensor to the sample
        var prevSample = TensorHelper.AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());

        return prevSample;
    }

    public override DenseTensor<float> Step(Tensor<float> modelOutput, int timestep, Tensor<float> sample, int order = 4)
    {
        throw new NotSupportedException("Please use LMSDiscreteSchedulerFloat");
    }

    public override Tensor<Float16> ScaleModelInput(Tensor<Float16> sample, int timestep)
    {
        // Get step index of timestep from TimeSteps
        int stepIndex = Timesteps.IndexOf(timestep);

        // Get sigma at stepIndex
        var sigma = Sigmas[stepIndex];
        sigma = (Float16)Math.Sqrt(Math.Pow(sigma, 2) + 1);

        // Divide sample tensor shape by sigma
        sample = TensorHelper.DivideTensorByFloat(sample.ToArray(), (Float16)sigma, sample.Dimensions.ToArray());

        _isScaleInputCalled = true;

        return sample;
    }

    public override Tensor<float> ScaleModelInput(Tensor<float> sample, int timestep)
    {
        throw new NotSupportedException("Please use LMSDiscreteScheduler");
    }
}
