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

using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

//public class EulerAncestralDiscreteScheduler : SchedulerBase
//{
//    private readonly string _predictionType;
//    public override float InitNoiseSigma { get; protected set; }
//    public int numInferenceSteps;
//    public override List<int> Timesteps { get; protected set; }
//    public override Tensor<float> Sigmas { get; protected set; }

//    public EulerAncestralDiscreteScheduler(int numTrainTimesteps = 1000, float betaStart = 0.00085f, float betaEnd = 0.012f, string betaSchedule = "scaled_linear", List<float>? trainedBetas = null, string prediction_type = "epsilon") : base(numTrainTimesteps)
//    {
//        var alphas = new List<float>();
//        var betas = new List<float>();
//        _predictionType = prediction_type;

//        if (trainedBetas != null)
//        {
//            betas = trainedBetas;
//        }
//        else if (betaSchedule == "linear")
//        {
//            betas = Enumerable.Range(0, numTrainTimesteps).Select(i => betaStart + (betaEnd - betaStart) * i / (numTrainTimesteps - 1)).ToList();
//        }
//        else if (betaSchedule == "scaled_linear")
//        {
//            var start = (float)Math.Sqrt(betaStart);
//            var end = (float)Math.Sqrt(betaEnd);
//            betas = np.linspace(start, end, numTrainTimesteps).ToArray<float>().Select(x => x * x).ToList();
//        }
//        else
//        {
//            throw new Exception("betaSchedule must be one of 'linear' or 'scaled_linear'");
//        }

//        alphas = betas.Select(beta => 1 - beta).ToList();

//        _alphasCumulativeProducts = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b)).ToList();

//        // Create sigmas as a list and reverse it
//        var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();

//        // Standard deviation of the initial noise distrubution
//        InitNoiseSigma = (float)sigmas.Max();
//    }

//    public override void SetTimesteps(int num_inference_steps)
//    {
//        double start = 0;
//        double stop = _numTrainTimesteps - 1;
//        double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

//        Timesteps = timesteps.Select(x => (int)x).Reverse().ToList();

//        var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
//        var range = np.arange(0, (double)sigmas.Count).ToArray<double>();
//        sigmas = Interpolate(timesteps, range, sigmas).ToList();
//        InitNoiseSigma = (float)sigmas.Max();
//        Sigmas = new DenseTensor<float>(sigmas.Count());
//        for (int i = 0; i < sigmas.Count(); i++)
//        {
//            Sigmas[i] = (float)sigmas[i];
//        }
//    }

//    public override DenseTensor<float> Step(Tensor<float> modelOutput, int timestep, Tensor<float> sample, int order = 4)
//    {
//        if (!_isScaleInputCalled)
//        {
//            Console.WriteLine("The `scale_model_input` function should be called before `step` to ensure correct denoising. See `StableDiffusionPipeline` for a usage example.");
//        }

//        int stepIndex = Timesteps.IndexOf(timestep);
//        var sigma = Sigmas[stepIndex];

//        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
//        Tensor<float>? predOriginalSample = null;
//        if (_predictionType == "epsilon")
//        {
//            //  pred_original_sample = sample - sigma * model_output
//            predOriginalSample = TensorHelper.SubtractTensors(sample, TensorHelper.MultipleTensorByFloat(modelOutput, sigma));
//        }
//        else if (_predictionType == "v_prediction")
//        {
//            // * c_out + input * c_skip
//            //predOriginalSample = modelOutput * (-sigma / Math.Pow(sigma * sigma + 1, 0.5)) + (sample / (sigma * sigma + 1));
//            throw new NotImplementedException($"predictionType not implemented yet: {_predictionType}");
//        }
//        else if (_predictionType == "sample")
//        {
//            throw new NotImplementedException($"predictionType not implemented yet: {_predictionType}");
//        }
//        else
//        {
//            throw new ArgumentException($"predictionType given as {_predictionType} must be one of `epsilon`, or `v_prediction`");
//        }

//        float sigmaFrom = Sigmas[stepIndex];
//        float sigmaTo = Sigmas[stepIndex + 1];

//        var sigmaFromLessSigmaTo = MathF.Pow(sigmaFrom, 2) - MathF.Pow(sigmaTo, 2);
//        var sigmaUpResult = MathF.Pow(sigmaTo, 2) * sigmaFromLessSigmaTo / MathF.Pow(sigmaFrom, 2);
//        var sigmaUp = sigmaUpResult < 0 ? -MathF.Pow(MathF.Abs(sigmaUpResult), 0.5f) : MathF.Pow(sigmaUpResult, 0.5f);

//        var sigmaDownResult = MathF.Pow(sigmaTo, 2) - MathF.Pow(sigmaUp, 2);
//        var sigmaDown = sigmaDownResult < 0 ? -MathF.Pow(MathF.Abs(sigmaDownResult), 0.5f) : MathF.Pow(sigmaDownResult, 0.5f);

//        // 2. Convert to an ODE derivative
//        var sampleMinusPredOriginalSample = TensorHelper.SubtractTensors(sample, predOriginalSample);
//        DenseTensor<float> derivative = TensorHelper.DivideTensorByFloat(sampleMinusPredOriginalSample.ToArray(), sigma, predOriginalSample.Dimensions.ToArray());// (sample - predOriginalSample) / sigma;

//        float dt = sigmaDown - sigma;

//        DenseTensor<float> prevSample = TensorHelper.AddTensors(sample, TensorHelper.MultipleTensorByFloat(derivative, dt));// sample + derivative * dt;

//        //var noise = generator == null ? np.random.randn(modelOutput.shape) : np.random.RandomState(generator).randn(modelOutput.shape);
//        var noise = TensorHelper.GetRandomTensor(prevSample.Dimensions);

//        var noiseSigmaUpProduct = TensorHelper.MultipleTensorByFloat(noise, sigmaUp);
//        prevSample = TensorHelper.AddTensors(prevSample, noiseSigmaUpProduct);// prevSample + noise * sigmaUp;

//        return prevSample;
//    }

//    public override void Dispose()
//    {
//    }
//}
