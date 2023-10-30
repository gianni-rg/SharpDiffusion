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

// Ported/based on Hugging Face Diffusers library
// (https://github.com/huggingface/diffusers)
// Copyright (C) 2022-2023 The HuggingFace Inc. team.
// Licensed under the Apache License, Version 2.0.

namespace SharpDiffusion;

using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SharpDiffusion.Schedulers;
using System;
using System.Collections.Generic;
using System.IO;

public class OnnxStableDiffusionPipelineFloat16 : DiffusionPipeline
{
    private readonly ILogger _logger;
    private readonly OnnxRuntimeModel _vaeEncoder;
    private readonly OnnxRuntimeModel _vaeDecoder;
    private readonly OnnxRuntimeModel _textEncoder;
    private readonly OnnxRuntimeModel _tokenizer;
    private readonly OnnxRuntimeModel _unet;
    private readonly SchedulerType _schedulerType;
    private readonly SchedulerFactory _schedulerFactory;
    private readonly OnnxRuntimeModel _safetyChecker;

    public object textInputIds { get; private set; }

    public OnnxStableDiffusionPipelineFloat16(ILogger? logger, OnnxRuntimeModel vaeEncoder, OnnxRuntimeModel vaeDecoder, OnnxRuntimeModel textEncoder, OnnxRuntimeModel tokenizer, OnnxRuntimeModel unet, SchedulerType schedulerType, OnnxRuntimeModel safetyChecker, bool requiresSafetyChecker = true) : base()
    {
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<OnnxStableDiffusionPipeline>.Instance;
        _vaeEncoder = vaeEncoder;
        _vaeDecoder = vaeDecoder;
        _textEncoder = textEncoder;
        _tokenizer = tokenizer;
        _unet = unet;
        _safetyChecker = safetyChecker;

        _schedulerType = schedulerType;
        _schedulerFactory = new SchedulerFactory();

        if (_safetyChecker is null && requiresSafetyChecker)
        {
            _logger.LogWarning($"You have disabled the safety checker for OnnxStableDiffusionPipeline by passing 'safety_checker = null'. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered" +
               " results in services or applications open to the public. Both the diffusers team and Hugging Face" +
               " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling" +
               " it only for use-cases that involve analyzing network behavior or auditing its results. For more" +
               " information, please have a look at https://github.com/huggingface/diffusers/pull/254.");
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _vaeEncoder.Dispose();
                _vaeDecoder.Dispose();
                _textEncoder.Dispose();
                _tokenizer.Dispose();
                _unet.Dispose();
                _safetyChecker.Dispose();
            }

            // TODO: free unmanaged resources (unmanaged objects) and override finalizer
            // TODO: set large fields to null
            _disposed = true;
        }
    }

    ///// <summary>
    ///// Instantiate a Diffusion pipeline from pre-trained pipeline weights
    ///// </summary>
    //public static IDiffusionPipeline FromPretrained(string pretrainedModelNameOrPath, string? provider = null, Dictionary<string, string>? sessionOptions = null)
    //{
    //    //TODO: in the caller, create session options for custom op of extensions
    //    //sessionOptions.Add("OrtExtensionsPath", "");

    //    var pipeline = DiffusionPipelineFactory.FromPretrained<OnnxStableDiffusionPipeline<Float16>>(
    //        pretrainedModelNameOrPath,
    //        provider: provider,
    //        sessionOptions: sessionOptions);

    //    return pipeline ?? throw new InvalidOperationException("Unable to load the OnnxStableDiffusionPipeline");
    //}


    public override StableDiffusionPipelineOutput Run(List<string> prompts, List<string> negativePrompts, StableDiffusionConfig config, Action<int>? callback = null)
    {
        var batchSize = prompts.Count;

        if (config.Height % 8 != 0 || config.Width % 8 != 0)
        {
            throw new InvalidDataException($"'height' and 'width' have to be divisible by 8 but are {config.Height} and {config.Width}.");
        }

        if (config.CallbackSteps <= 0)
        {
            throw new ArgumentOutOfRangeException("'callbackSteps' has to be a positive integer.");
        }

        var seed = config.Seed ?? new Random().Next();
        var generator = new Random(seed);

        var doClassifierFreeGuidance = config.GuidanceScale > 1.0f;

        var textEmbeddings = EncodePrompts(prompts, doClassifierFreeGuidance, negativePrompts, config);

        //TODO: Need to recreate for each execution, as the scheduler is stateful (!?); to be investigated
        var scheduler = _schedulerFactory.GetScheduler<Float16>(_schedulerType);

        // Get the initial random noise (unless the user supplied it)
        var latents = GenerateLatentSamples(batchSize, generator, config, scheduler.InitNoiseSigma);

        // Set timesteps
        scheduler.SetTimesteps(config.NumInferenceSteps);

        for (int t = 0; t < scheduler.Timesteps.Count; t++)
        {
            // Expand the latents if we are doing classifier free guidance
            var latentModelInput = doClassifierFreeGuidance ? TensorHelpers.Duplicate(latents.ToArray(), new[] { 2 * batchSize * config.NumImagesPerPrompt, config.Channels, config.Height / 8, config.Width / 8 }) : latents;
            latentModelInput = scheduler.ScaleModelInput(latentModelInput, scheduler.Timesteps[t]);

            // Predict the noise residual
            //var noisePred = _unet.Score(sample = latentModelInput, timestep = _scheduler.Timesteps[t], encoderHiddenStates = textEmbeddings);
            var input = CreateUnetModelInput(textEmbeddings, latentModelInput, scheduler.Timesteps[t]);

            // Run Inference
            var unetOutput = _unet.Run(input);
            //noisePred = noisePred[0];
            var noisePred = unetOutput.First().AsTensor<Float16>();
            if (noisePred is null)
            {
                throw new InvalidOperationException("Unable to execute UNET inference (noise prediction is null)");
            }

            // Perform guidance
            if (doClassifierFreeGuidance)
            {
                // Split tensors from 2 * (batchSize * config.NumImagesPerPrompt),4,64,64 to 1 * (batchSize * config.NumImagesPerPrompt),4,64,64
                //noisePredUncond, noisePredText = np.split(noisePred, 2);
                //var (noisePredUncond, noisePredText) = TensorHelper.SplitTensor(noisePred, new[] { batchSize * config.NumImagesPerPrompt, config.Channels, config.Height / 8, config.Width / 8 });
                var (noisePredUncond, noisePredText) = TensorHelpers.SplitTensor(noisePred, 2);
                //noisePred = noisePredUncond + guidanceScale * (noisePredText - noisePredUncond);
                noisePred = PerformGuidance(noisePredUncond, noisePredText, config.GuidanceScale);
            }

            // Compute the previous noisy sample x_t -> x_t-1
            //scheduler_output = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs);
            //latents = scheduler_output.prev_sample.numpy();
            latents = scheduler.Step(noisePred, scheduler.Timesteps[t], latents);

            // Call the callback, if provided
            if (callback is not null && t % config.CallbackSteps == 0)
            {
                //callback(t, latents);
                callback(t);
            }
        }

        // Scale and decode the image latents with VAE
        //latents = 1 / 0.18215 * latents;
        latents = TensorHelpers.MultipleTensorByFloat(latents.ToArray(), (Float16)(1.0f / 0.18215f), latents.Dimensions.ToArray());

        // Decode image(s)
        //# image = self.vae_decoder(latent_sample=latents)[0]
        //# it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        //image = np.concatenate([_vaeDecoder(latent_sample: latents[i: i + 1])[0] for i in range(latents.shape[0])])
        var resultTensors = new List<Tensor<Float16>>();
        for (int i = 0; i < latents.Dimensions[0]; i++)
        {
            //image = _vaeDecoder(latent_sample: latents[i: i + 1])[0];
            //image = np.clip(image / 2 + 0.5, 0, 1);
            //image = image.transpose((0, 2, 3, 1)); // swap channels (BCHW -> BHWC)
            var decoderInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("latent_sample", TensorHelpers.CreateTensor((latents as DenseTensor<Float16>)!.Buffer.Slice(i * latents.Strides[0], latents.Strides[0]).ToArray(), new[] { 1, config.Channels, config.Height / 8, config.Width / 8 })) };
            var decoderOutput = _vaeDecoder.Run(decoderInput);
            var imageResultTensor = decoderOutput.First().AsTensor<Float16>();
            resultTensors.Add(imageResultTensor!);
        }


        //TODO: to be fully ported from Python
        // see: https://docs.sixlabors.com/articles/imagesharp/pixelbuffers.html
        // Rgba32[] pixelArray = new Rgba32[image.Width * image.Height]
        // image.CopyPixelDataTo(pixelArray);
        //byte[] rgbaBytes = GetMyRgbaBytes();
        //using (var image = Image.LoadPixelData<Rgba32>(rgbaBytes, width, height))
        //{
        //    // Work with the image
        //}

        //if (outputType == "image")
        //{
        var images = ConvertToImages(resultTensors, config);
        //}

        List<bool> hasNsfwConcept = new List<bool>();
        if (_safetyChecker is not null)
        {
            foreach (var image in images)
            {
                // CLIP input (BCHW), Image input (BHWC)
                var (inputTensor, inputImageTensor) = PreprocessImageForCLIPFeatureExtractor(image);

                // There will throw an error if use SafetyChecker batch size > 1
                var safetyCheckerInput = new List<NamedOnnxValue> {
                    NamedOnnxValue.CreateFromTensor("clip_input", inputTensor),
                    NamedOnnxValue.CreateFromTensor("images", inputImageTensor)
                };

                var safetyCheckerOutput = _safetyChecker.Run(safetyCheckerInput);
                var result = safetyCheckerOutput.Last().AsEnumerable<bool>().First();
                hasNsfwConcept.Add(result);
            }
        }

        scheduler.Dispose();

        return new StableDiffusionPipelineOutput(images: images, nsfwContentDetected: hasNsfwConcept);
    }

    private (Tensor<Float16>, Tensor<Float16>) PreprocessImageForCLIPFeatureExtractor(Image<Rgba32> image)
    {
        // Resize image to 224x224
        var scaledImage = image.Clone(x =>
        {
            x.Resize(new ResizeOptions
            {
                Size = new Size(224, 224),
                Mode = ResizeMode.Crop
            });
        });
        
        // Preprocess image
        var inputTensor = new DenseTensor<Float16>(new[] { 1, 3, 224, 224 });
        var inputImageTensor = new DenseTensor<Float16>(new[] { 1, 224, 224, 3 });
        var mean = new[] { 0.485f, 0.456f, 0.406f };
        var stddev = new[] { 0.229f, 0.224f, 0.225f };
        scaledImage.ProcessPixelRows(pixelAccessor =>
        {
            for (int y = 0; y < pixelAccessor.Height; y++)
            {
                Span<Rgba32> pixelSpan = pixelAccessor.GetRowSpan(y);

                for (int x = 0; x < pixelSpan.Length; x++)
                {
                    inputTensor[0, 0, y, x] = (Float16)(((pixelSpan[x].R / 255f) - mean[0]) / stddev[0]);
                    inputTensor[0, 1, y, x] = (Float16)(((pixelSpan[x].G / 255f) - mean[1]) / stddev[1]);
                    inputTensor[0, 2, y, x] = (Float16)(((pixelSpan[x].B / 255f) - mean[2]) / stddev[2]);
                    inputImageTensor[0, y, x, 0] = inputTensor[0, 0, y, x];
                    inputImageTensor[0, y, x, 1] = inputTensor[0, 1, y, x];
                    inputImageTensor[0, y, x, 2] = inputTensor[0, 2, y, x];
                }
            }
        });

        return (inputTensor, inputImageTensor);
    }

    /// <summary>
    /// Encodes the prompt into text encoder hidden states
    /// </summary>
    /// <param name="prompts">Prompt(s) to be encoded</param>
    /// <param name="numImagesPerPrompt">number of images that should be generated per prompt</param>
    /// <param name="doClassifierFreeGuidance">whether to use classifier free guidance or not</param>
    /// <param name="negativePrompts">Prompt(s) not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).</param>
    private DenseTensor<Float16> EncodePrompts(List<string> prompts, bool doClassifierFreeGuidance, List<string>? negativePrompts, StableDiffusionConfig config)
    {
        int batchSize = prompts.Count;

        // Get prompt text embeddings
        //var textInputs = _tokenizer.Run(prompts, padding: "max_length", max_length: _tokenizer.ModelMaxLength, truncation: true, returnTensors: "np");
        //var textInputIds = textInputs.InputIds;
        var textInputsIds = TokenizeText(prompts, config);

        //var untruncatedIds = _tokenizer.Run(prompts, padding: "max_length", returnTensors: "np").input_ids;
        //if (!np.array_equal(text_input_ids, untruncated_ids))
        //{
        //    var removedText = _tokenizer.batchDecode(untruncated_ids[:, _tokenizer.ModelMaxLength - 1 : -1])
        //    //_logger.LogWarning($"The following part of your input was truncated because CLIP can only handle sequences up to {_tokenizer.ModelMaxLength} tokens: {removedText}");
        //}

        //var textEmbeddings = _textEncoder.Run(textInputIds)[0];
        var textEmbeddings = EncodeText(textInputsIds, config);
        //textEmbeddings = np.repeat(textEmbeddings, numImagesPerPrompt, axis: 0);
        textEmbeddings = TensorHelpers.Repeat(textEmbeddings, config.NumImagesPerPrompt);

        // Get unconditional embeddings for classifier free guidance
        if (doClassifierFreeGuidance)
        {
            List<string> uncondTokens;
            if (negativePrompts is null)
            {
                uncondTokens = new List<string>(batchSize);
                for (int i = 0; i < batchSize; i++)
                {
                    uncondTokens.Add(string.Empty);
                }
            }
            else if (negativePrompts.Count == 1)
            {
                uncondTokens = new List<string>(batchSize);
                for (int i = 0; i < batchSize; i++)
                {
                    uncondTokens.Add(negativePrompts[0]);
                }
            }
            else if (batchSize != negativePrompts.Count)
            {
                throw new InvalidDataException($"'negativePrompts' has batch size {negativePrompts.Count}, but 'prompt' has batch size {prompts.Count}. Please make sure that passed 'negativePrompts' matches the batch size of 'prompts'");
            }
            else
            {
                uncondTokens = negativePrompts;
            }

            //var maxLength = textInputIds.shape[-1];
            //var uncond_input = _tokenizer(uncond_tokens, padding: "max_length", max_length: maxLength, truncation: true, returnTensors = "np");
            var uncondInput = TokenizeText(uncondTokens, config);
            //var uncondInput = CreateUncondInput(batchSize, config);
            //var uncondEmbeddings = _textEncoder(input_ids: uncond_input.input_ids.astype(np.int32))[0];
            var uncondEmbeddings = EncodeText(uncondInput, config);
            //uncondEmbeddings = np.repeat(uncondEmbeddings, numImagesPerPrompt, axis: 0);
            uncondEmbeddings = TensorHelpers.Repeat(uncondEmbeddings, config.NumImagesPerPrompt);

            // For classifier free guidance, we need to do two forward passes.
            // Here we concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes.
            //text_embeddings = np.concatenate([uncond_embeddings, text_embeddings]);
            var textEmbeddingsTensor = TensorHelpers.Concatenate(uncondEmbeddings.ToArray(), textEmbeddings.ToArray(), new[] { 2 * batchSize * config.NumImagesPerPrompt, config.TokenizerModelMaxLength, config.ModelEmbeddingSize });

            //DenseTensor<float> textEmbeddingsTensor = new DenseTensor<float>(new[] { 2 * batchSize * config.NumImagesPerPrompt, config.TokenizerModelMaxLength, config.ModelEmbeddingSize });
            //for (var i = 0; i < textEmbeddings.Length; i++)
            //{
            //    textEmbeddingsTensor[0, i / config.ModelEmbeddingSize, i % config.ModelEmbeddingSize] = uncondEmbedding[i];
            //    textEmbeddingsTensor[1, i / config.ModelEmbeddingSize, i % config.ModelEmbeddingSize] = textEmbeddings[i];
            //}

            return textEmbeddingsTensor;
        }
        else
        {
            return TensorHelpers.CreateTensor(textEmbeddings.ToArray(), new[] { batchSize * config.NumImagesPerPrompt, config.TokenizerModelMaxLength, config.ModelEmbeddingSize });
        }
    }


    private static Tensor<Float16> GenerateLatentSamples(int batchSize, Random generator, StableDiffusionConfig config, float initNoiseSigma)
    {
        var latents = new DenseTensor<Float16>(new[] { batchSize * config.NumImagesPerPrompt, config.Channels, config.Height / 8, config.Width / 8 });
        var latentsArray = latents.ToArray();

        for (int i = 0; i < latentsArray.Length; i++)
        {
            // Generate a random number from a normal distribution with mean 0 and variance 1
            var u1 = generator.NextDouble();                   // Uniform(0,1) random number
            var u2 = generator.NextDouble();                   // Uniform(0,1) random number
            var radius = Math.Sqrt(-2.0 * Math.Log(u1));       // Radius of polar coordinates
            var theta = 2.0 * Math.PI * u2;                    // Angle of polar coordinates
            var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number

            // Add noise to latents (scaled by scheduler.InitNoiseSigma)
            // Generate randoms that are negative and positive
            latentsArray[i] = (Float16)((float)standardNormalRand * initNoiseSigma);
        }

        latents = TensorHelpers.CreateTensor(latentsArray, latents.Dimensions.ToArray());

        return latents;
    }

    private static List<NamedOnnxValue> CreateUnetModelInput(Tensor<Float16> encoderHiddenStates, Tensor<Float16> sample, long timeStep)
    {
        var input = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("sample", sample),
                NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<Float16>(new Float16[] { (Float16)timeStep }, new int[] { 1 }))
            };

        return input;
    }

    private static Tensor<Float16> PerformGuidance(Tensor<Float16> noisePred, Tensor<Float16> noisePredText, double guidanceScale)
    {
        for (int i = 0; i < noisePred.Dimensions[0]; i++)
        {
            for (int j = 0; j < noisePred.Dimensions[1]; j++)
            {
                for (int k = 0; k < noisePred.Dimensions[2]; k++)
                {
                    for (int l = 0; l < noisePred.Dimensions[3]; l++)
                    {
                        var noisePredFloat = noisePred[i, j, k, l];
                        var noisePredTextFloat = noisePredText[i, j, k, l];
                        noisePred[i, j, k, l] = noisePredFloat.Add(noisePredTextFloat.Subtract(noisePredFloat).Mul((Float16)guidanceScale));
                    }
                }
            }
        }
        return noisePred;
    }

    public List<int[]> TokenizeText(List<string> text, StableDiffusionConfig config)
    {
        var batchedInputIdsInt = new List<int[]>();

        var inputTensor = new DenseTensor<string>(text.ToArray(), new int[] { text.Count });
        var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("string_input", inputTensor) };

        var tokens = _tokenizer.Run(inputString);

        var ids = tokens.First().Value as DenseTensor<long>;

        for (int i = 0; i < ids.Dimensions[0]; i++)
        {
            // TRIM input text to modelMaxLength
            var inputLength = Math.Min(ids.Strides[0], config.TokenizerModelMaxLength);
            var inputIds = new long[inputLength];
            ids.Buffer.Slice(i * ids.Strides[0], inputLength).CopyTo(inputIds);

            // Cast inputIds to Int32
            var inputIdsInt = inputIds?.Select(x => (int)x).ToArray();

            // Pad array with 49407 until length is modelMaxLength
            if (inputIdsInt?.Length < config.TokenizerModelMaxLength)
            {
                var pad = Enumerable.Repeat(49407, config.TokenizerModelMaxLength - inputIdsInt.Length).ToArray();
                inputIdsInt = inputIdsInt.Concat(pad).ToArray();
            }
            else
            {
                inputIdsInt[config.TokenizerModelMaxLength - 1] = 49407; // EOT
            }


            batchedInputIdsInt.Add(inputIdsInt);
        }
        return batchedInputIdsInt;
    }

    public DenseTensor<Float16> EncodeText(List<int[]> tokenizedInputs, StableDiffusionConfig config)
    {
        // Create input tensor
        var input_ids = TensorHelpers.CreateTensor(tokenizedInputs.SelectMany(l => l).ToArray(), new[] { tokenizedInputs.Count, config.TokenizerModelMaxLength });
        var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids) };

        var encoded = _textEncoder.Run(input);

        var lastHiddenState = encoded.First().AsEnumerable<Float16>();
        var lastHiddenStateTensor = TensorHelpers.CreateTensor(lastHiddenState.ToArray(), new[] { tokenizedInputs.Count, config.TokenizerModelMaxLength, config.ModelEmbeddingSize });

        return lastHiddenStateTensor;
    }

    //float CreateSingle(bool sign, byte exp, uint sig) => BitConverter.UInt32BitsToSingle(((sign ? 1U : 0U) << 31) + ((uint)exp << 23) + sig);
    //double CreateDouble(bool sign, ushort exp, ulong sig) => BitConverter.UInt64BitsToDouble(((sign ? 1UL : 0UL) << 63) + ((ulong)exp << 52) + sig);

    //static string FloatToBinary(float f)
    //{
    //    StringBuilder sb = new StringBuilder();
    //    Byte[] ba = BitConverter.GetBytes(f);
    //    foreach (Byte b in ba)
    //        for (int i = 0; i < 8; i++)
    //        {
    //            sb.Insert(0, ((b >> i) & 1) == 1 ? "1" : "0");
    //        }
    //    string s = sb.ToString();
    //    string r = s.Substring(0, 1) + " " + s.Substring(1, 8) + " " + s.Substring(9); //sign exponent mantissa
    //    return r;
    //}

    //float CreateSingle(ushort n)
    //{
    //    Debug.Assert(n >= 0 && n <= 65535);
    //    var sign = n >> 15;
    //    var exp = (n >> 10) & 0b011111;
    //    var fraction = n & (2^10 - 1);
    //    if (exp == 0)
    //    {
    //        if (fraction == 0)
    //        {
    //            return sign == 1 ? -0.0f : 0.0f;
    //        }
    //        else
    //        {
    //            return (-1)^sign * fraction / 2^10 * 2^(-14);  // subnormal
    //        }
    //    }
    //    else if (exp == 0b11111)
    //    {
    //        if (fraction == 0)
    //        {
    //            return sign == 1 ? float.NegativeInfinity : float.PositiveInfinity;
    //        }
    //    }
    //    else
    //    {
    //        return float.NaN;
    //    }

    //    return (-1)^sign * (1 + fraction / 2^10) * 2^(exp - 15);
    //}

    public static List<int[]> CreateUncondInput(int batchSize, StableDiffusionConfig config)
    {
        // Create an array of empty tokens for the unconditional input.
        var blankTokenValue = 49407;

        var batchedInputIds = new List<int[]>();
        for (int i = 0; i < batchSize; i++)
        {
            var inputIds = new List<int>
            {
                49406
            };

            var pad = Enumerable.Repeat(blankTokenValue, config.TokenizerModelMaxLength - inputIds.Count).ToArray();
            inputIds.AddRange(pad);

            batchedInputIds.Add(inputIds.ToArray());
        }
        return batchedInputIds;
    }

    public static List<Image<Rgba32>> ConvertToImages(List<Tensor<Float16>> output, StableDiffusionConfig config)
    {
        var results = new List<Image<Rgba32>>();
        for (int i = 0; i < output.Count; i++)
        {
            var result = new Image<Rgba32>(config.Width, config.Height);

            for (var y = 0; y < config.Height; y++)
            {
                for (var x = 0; x < config.Width; x++)
                {
                    result[x, y] = new Rgba32(
                        (byte)Math.Round(Math.Clamp(((float)output[i][0, 0, y, x]) / 2 + 0.5, 0, 1) * 255),
                        (byte)Math.Round(Math.Clamp(((float)output[i][0, 1, y, x]) / 2 + 0.5, 0, 1) * 255),
                        (byte)Math.Round(Math.Clamp(((float)output[i][0, 2, y, x]) / 2 + 0.5, 0, 1) * 255)
                    );
                }
            }

            results.Add(result);
        }

        return results;
    }
}