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

using SharpDiffusion.Interfaces;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using SharpDiffusion.Schedulers;

public class OnnxStableDiffusionPipeline : DiffusionPipeline
{
    private readonly ILogger _logger;
    private readonly OnnxRuntimeModel _vaeEncoder;
    private readonly OnnxRuntimeModel _vaeDecoder;
    private readonly OnnxRuntimeModel _textEncoder;
    private readonly OnnxRuntimeModel _tokenizer;
    private readonly OnnxRuntimeModel _unet;
    private IScheduler _scheduler;
    private readonly OnnxRuntimeModel _safetyChecker;

    public object textInputIds { get; private set; }

    public OnnxStableDiffusionPipeline(ILogger? logger, OnnxRuntimeModel vaeEncoder, OnnxRuntimeModel vaeDecoder, OnnxRuntimeModel textEncoder, OnnxRuntimeModel tokenizer, OnnxRuntimeModel unet, IScheduler scheduler, OnnxRuntimeModel safetyChecker, bool requiresSafetyChecker = true) : base()
    {
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<OnnxStableDiffusionPipeline>.Instance;
        _vaeEncoder = vaeEncoder;
        _vaeDecoder = vaeDecoder;
        _textEncoder = textEncoder;
        _tokenizer = tokenizer;
        _unet = unet;
        _scheduler = scheduler;
        _safetyChecker = safetyChecker;

        if (_safetyChecker is null && requiresSafetyChecker)
        {
            _logger.LogWarning($"You have disabled the safety checker for OnnxStableDiffusionPipeline by passing 'safety_checker = null'. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered" +
               " results in services or applications open to the public. Both the diffusers team and Hugging Face" +
               " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling" +
               " it only for use-cases that involve analyzing network behavior or auditing its results. For more" +
               " information, please have a look at https://github.com/huggingface/diffusers/pull/254.");
        }
    }

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

        //TODO: Need to recreate for each execution, as the scheduler is stateful (!?), investigate
        _scheduler = new LMSDiscreteScheduler();

        // Get the initial random noise (unless the user supplied it)
        var latents = GenerateLatentSamples(batchSize, generator, config, _scheduler.InitNoiseSigma);

        // Set timesteps
        _scheduler.SetTimesteps(config.NumInferenceSteps);

        var input = new List<NamedOnnxValue>();
        for (int t = 0; t < _scheduler.Timesteps.Count; t++)
        {
            // Expand the latents if we are doing classifier free guidance
            var latentModelInput = doClassifierFreeGuidance ? TensorHelper.Duplicate(latents.ToArray(), new[] { 2 * batchSize * config.NumImagesPerPrompt, config.Channels, config.Height / 8, config.Width / 8 }) : latents;
            latentModelInput = _scheduler.ScaleModelInput(latentModelInput, _scheduler.Timesteps[t]);

            // Predict the noise residual
            //var noisePred = _unet.Score(sample = latentModelInput, timestep = _scheduler.Timesteps[t], encoderHiddenStates = textEmbeddings);
            input = CreateUnetModelInput(textEmbeddings, latentModelInput, _scheduler.Timesteps[t]);

            // Run Inference
            var unetOutput = _unet.Run(input);
            //noisePred = noisePred[0];
            var noisePred = unetOutput.First().Value as Tensor<float>;

            // Perform guidance
            if (doClassifierFreeGuidance)
            {
                // Split tensors from 2 * (batchSize * config.NumImagesPerPrompt),4,64,64 to 1 * (batchSize * config.NumImagesPerPrompt),4,64,64
                //noisePredUncond, noisePredText = np.split(noisePred, 2);
                //var (noisePredUncond, noisePredText) = TensorHelper.SplitTensor(noisePred, new[] { batchSize * config.NumImagesPerPrompt, config.Channels, config.Height / 8, config.Width / 8 });
                var (noisePredUncond, noisePredText) = TensorHelper.SplitTensor(noisePred, 2);
                //noisePred = noisePredUncond + guidanceScale * (noisePredText - noisePredUncond);
                noisePred = PerformGuidance(noisePredUncond, noisePredText, config.GuidanceScale);
            }

            // Compute the previous noisy sample x_t -> x_t-1
            //scheduler_output = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs);
            //latents = scheduler_output.prev_sample.numpy();
            latents = _scheduler.Step(noisePred, _scheduler.Timesteps[t], latents);

            // Call the callback, if provided
            if (callback is not null && t % config.CallbackSteps == 0)
            {
                //callback(t, latents);
                callback(t);
            }
        }

        // Scale and decode the image latents with VAE
        //latents = 1 / 0.18215 * latents;
        latents = TensorHelper.MultipleTensorByFloat(latents.ToArray(), 1.0f / 0.18215f, latents.Dimensions.ToArray());

        // Decode image(s)
        //# image = self.vae_decoder(latent_sample=latents)[0]
        //# it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        //image = np.concatenate([_vaeDecoder(latent_sample: latents[i: i + 1])[0] for i in range(latents.shape[0])])
        var resultTensors = new List<Tensor<float>>();
        for (int i = 0; i < latents.Dimensions[0]; i++)
        {
            //image = _vaeDecoder(latent_sample: latents[i: i + 1])[0];
            //image = np.clip(image / 2 + 0.5, 0, 1);
            //image = image.transpose((0, 2, 3, 1)); // swap channels (BCHW -> BHWC)
            var decoderInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("latent_sample", TensorHelper.CreateTensor((latents as DenseTensor<float>)!.Buffer.Slice(i * latents.Strides[0], latents.Strides[0]).ToArray(), new[] { 1, config.Channels, config.Height / 8, config.Width / 8 })) };
            var decoderOutput = _vaeDecoder.Run(decoderInput);
            var imageResultTensor = decoderOutput.First().Value as Tensor<float>;
            resultTensors.Add(imageResultTensor!);
        }

        //var decoderInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("latent_sample", latents) };
        //var decoderOutput = _vaeDecoder.Run(decoderInput);
        //var imageResultTensor = decoderOutput.First().Value as Tensor<float>;

        // TODO: implement safety checker model
        List<bool[]>? hasNsfwConcept = null;
        if (_safetyChecker is not null)
        {
            //var safetyCheckerInput = _featureExtractor(numpy_to_pil(image), return_tensors: "np").pixel_values.astype(image.dtype);
            //image, has_nsfw_concepts = _safetyChecker.Run(clip_input: safety_checker_input, images: image);

            //// There will throw an error if use safety_checker batchsize>1
            //var images = new List<object>();
            //hasNsfwConcept = new List<object>();
            //for (int i = 0; i < image.Shape[0]; i++)
            //{
            //    image_i, has_nsfw_concept_i = _SafetyChecker(clip_input: safety_checker_input[i: i + 1], images: image[i: i + 1]);
            //    images.Add(image_i);
            //    has_nsfw_concept.Add(has_nsfw_concept_i);
            //}
            //image = np.concatenate(images);
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

        return new StableDiffusionPipelineOutput(images: images, nsfwContentDetected: hasNsfwConcept);
    }


    /// <summary>
    /// Instantiate a Diffusion pipeline from pre-trained pipeline weights
    /// </summary>
    public static OnnxStableDiffusionPipeline FromPretrained(string pretrainedModelNameOrPath, string? provider = null, Dictionary<string, string>? sessionOptions = null)
    {
        //TODO: in the caller, create session options for custom op of extensions
        //sessionOptions.Add("OrtExtensionsPath", "");

        var pipeline = DiffusionPipelineFactory.FromPretrained<OnnxStableDiffusionPipeline>(
            pretrainedModelNameOrPath,
            provider: provider,
            sessionOptions: sessionOptions);

        return pipeline ?? throw new InvalidOperationException("Unable to load the OnnxStableDiffusionPipeline");
    }

    /// <summary>
    /// Encodes the prompt into text encoder hidden states
    /// </summary>
    /// <param name="prompts">Prompt(s) to be encoded</param>
    /// <param name="numImagesPerPrompt">number of images that should be generated per prompt</param>
    /// <param name="doClassifierFreeGuidance">whether to use classifier free guidance or not</param>
    /// <param name="negativePrompts">Prompt(s) not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).</param>
    private DenseTensor<float> EncodePrompts(List<string> prompts, bool doClassifierFreeGuidance, List<string>? negativePrompts, StableDiffusionConfig config)
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
        textEmbeddings = TensorHelper.Repeat(textEmbeddings, config.NumImagesPerPrompt);

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
            uncondEmbeddings = TensorHelper.Repeat(uncondEmbeddings, config.NumImagesPerPrompt);

            // For classifier free guidance, we need to do two forward passes.
            // Here we concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes.
            //text_embeddings = np.concatenate([uncond_embeddings, text_embeddings]);
            var textEmbeddingsTensor = TensorHelper.Concatenate(uncondEmbeddings.ToArray(), textEmbeddings.ToArray(), new[] { 2 * batchSize * config.NumImagesPerPrompt, config.TokenizerModelMaxLength, config.ModelEmbeddingSize });
            
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
            return TensorHelper.CreateTensor(textEmbeddings.ToArray(), new[] { batchSize * config.NumImagesPerPrompt, config.TokenizerModelMaxLength, config.ModelEmbeddingSize });
        }
    }


    private static Tensor<float> GenerateLatentSamples(int batchSize, Random generator, StableDiffusionConfig config, float initNoiseSigma)
    {
        var latents = new DenseTensor<float>(new[] { batchSize * config.NumImagesPerPrompt, config.Channels, config.Height / 8, config.Width / 8 });
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
            latentsArray[i] = (float)standardNormalRand * initNoiseSigma;
        }

        latents = TensorHelper.CreateTensor(latentsArray, latents.Dimensions.ToArray());

        return latents;
    }

    private static List<NamedOnnxValue> CreateUnetModelInput(Tensor<float> encoderHiddenStates, Tensor<float> sample, long timeStep)
    {
        var input = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("sample", sample),
                NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<float>(new float[] { timeStep }, new int[] { 1 }))
            };

        return input;
    }

    private static Tensor<float> PerformGuidance(Tensor<float> noisePred, Tensor<float> noisePredText, double guidanceScale)
    {
        for (int i = 0; i < noisePred.Dimensions[0]; i++)
        {
            for (int j = 0; j < noisePred.Dimensions[1]; j++)
            {
                for (int k = 0; k < noisePred.Dimensions[2]; k++)
                {
                    for (int l = 0; l < noisePred.Dimensions[3]; l++)
                    {
                        noisePred[i, j, k, l] = noisePred[i, j, k, l] + (float)guidanceScale * (noisePredText[i, j, k, l] - noisePred[i, j, k, l]);
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

    public DenseTensor<float> EncodeText(List<int[]> tokenizedInputs, StableDiffusionConfig config)
    {
        // Create input tensor
        var input_ids = TensorHelper.CreateTensor(tokenizedInputs.SelectMany(l => l).ToArray(), new[] { tokenizedInputs.Count, config.TokenizerModelMaxLength });
        var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<int>("input_ids", input_ids) };

        var encoded = _textEncoder.Run(input);

        var lastHiddenState = encoded.First().Value as IEnumerable<float>;
        var lastHiddenStateTensor = TensorHelper.CreateTensor(lastHiddenState!.ToArray(), new[] { tokenizedInputs.Count, config.TokenizerModelMaxLength, config.ModelEmbeddingSize });

        return lastHiddenStateTensor;
    }

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

            var pad = Enumerable.Repeat(blankTokenValue, config.TokenizerModelMaxLength - inputIds.Count()).ToArray();
            inputIds.AddRange(pad);

            batchedInputIds.Add(inputIds.ToArray());
        }
        return batchedInputIds;
    }

    public static List<Image<Rgba32>> ConvertToImages(List<Tensor<float>> output, StableDiffusionConfig config)
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
                        (byte)Math.Round(Math.Clamp(output[i][0, 0, y, x] / 2 + 0.5, 0, 1) * 255),
                        (byte)Math.Round(Math.Clamp(output[i][0, 1, y, x] / 2 + 0.5, 0, 1) * 255),
                        (byte)Math.Round(Math.Clamp(output[i][0, 2, y, x] / 2 + 0.5, 0, 1) * 255)
                    );
                }
            }

            results.Add(result);
        }

        return results;
    }
}