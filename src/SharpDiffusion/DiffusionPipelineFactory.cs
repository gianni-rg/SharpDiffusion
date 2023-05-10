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
using SharpDiffusion.Schedulers;
using System.Text.Json;

public class DiffusionPipelineFactory
{
    /// <summary>
    /// Instantiate a Diffusion Pipeline from pre-trained pipeline weights
    /// </summary>
    public static T? FromPretrained<T>(string pretrainedModelNameOrPath, string? provider = null, Dictionary<string, string>? sessionOptions = null, ILoggerFactory? loggerFactory = null) where T : class
    {
        var configDict = DictFromJSONFile(Path.Join(pretrainedModelNameOrPath, DiffusionPipeline.MODEL_CONFIG_FILENAME));

        var type = typeof(T);
        return type switch
        {
            Type _ when type == typeof(OnnxStableDiffusionPipeline) => ConfigureOnnxStableDiffusionPipeline(configDict, pretrainedModelNameOrPath, provider, sessionOptions, loggerFactory) as T,
            _ => default
        };
    }

    private static OnnxStableDiffusionPipeline ConfigureOnnxStableDiffusionPipeline(Dictionary<string, object> config, string cachedFolder, string? provider, Dictionary<string, string>? sessionOptions, ILoggerFactory? loggerFactory = null)
    {
        var vaeEncoder = OnnxRuntimeModel.FromPretrained(Path.Join(cachedFolder, "vae_encoder"), provider: provider, sessionOptions: sessionOptions);
        var vaeDecoder = OnnxRuntimeModel.FromPretrained(Path.Join(cachedFolder, "vae_decoder"), provider: provider, sessionOptions: sessionOptions);
        var textEncoder = OnnxRuntimeModel.FromPretrained(Path.Join(cachedFolder, "text_encoder"), provider: provider, sessionOptions: sessionOptions);
        var unet = OnnxRuntimeModel.FromPretrained(Path.Join(cachedFolder, "unet"), provider: provider, sessionOptions: sessionOptions);
        var safetyChecker = OnnxRuntimeModel.FromPretrained(Path.Join(cachedFolder, "safety_checker"), provider: provider, sessionOptions: sessionOptions);
        var tokenizer = OnnxRuntimeModel.FromPretrained(Path.Join(cachedFolder, "tokenizer"), provider: provider, sessionOptions: sessionOptions, ortExtensionPath: ".\\runtimes\\win-x64\\native\\ortextensions.dll"); // TODO: improve configuration
        var schedulerType = SchedulerType.LMSDiscreteScheduler; // TODO: improve configuration

        return new OnnxStableDiffusionPipeline(loggerFactory?.CreateLogger<OnnxStableDiffusionPipeline>(), vaeEncoder, vaeDecoder, textEncoder, tokenizer, unet, schedulerType, safetyChecker, requiresSafetyChecker: false);
    }

    private static Dictionary<string, object> DictFromJSONFile(string configFile)
    {
        Dictionary<string, object>? configDict;

        try
        {
            using (var reader = new StreamReader(configFile))
            {
                var content = reader.ReadToEnd();
                configDict = JsonSerializer.Deserialize<Dictionary<string, object>>(content);
            }
        }
        catch (JsonException)
        {
            throw new InvalidOperationException($"It looks like the config file at '{configFile}' is not a valid JSON file.");
        }

        return configDict ?? new Dictionary<string, object>();
    }
}