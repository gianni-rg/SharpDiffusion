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

using Microsoft.ML.OnnxRuntime;

public class OnnxRuntimeModel : IDisposable
{
    private readonly InferenceSession _model;
    private readonly string? _modelSaveDir;
    private readonly string? _latestModelName;
    private bool _disposed;

    public OnnxRuntimeModel(InferenceSession model, string? modelSaveDir = null, string? latestModelName = null)
    {
        _model = model;
        _modelSaveDir = modelSaveDir;
        _latestModelName = latestModelName ?? DiffusionPipeline.ONNX_WEIGHTS_NAME;
    }

    /// <summary>
    /// Loads an ONNX Inference session with an ExecutionProvider. Default provider is <c>CPUExecutionProvider</c>.
    /// </summary>
	/// <param name="path">Directory from which to load</param>
	/// <param name="provider">ONNX Runtime execution provider to use for loading the model, defaults to <c>CPUExecutionProvider</c></param>
	/// <param name="sessionOptions"></param>
    private static InferenceSession LoadModel(string path, string? provider = null, Dictionary<string, string>? sessionOptions = null, string? ortExtensionsPath = null)
    {
        provider ??= "CPUExecutionProvider";

        SessionOptions options;
        
        if (provider == "CUDAExecutionProvider")
        {
            var providerOptions = new OrtCUDAProviderOptions();
            providerOptions.UpdateOptions(sessionOptions);
            options = SessionOptions.MakeSessionOptionWithCudaProvider(providerOptions);
        }
        else if (provider == "CPUExecutionProvider")
        {
            options = new SessionOptions();
        }
        else
        {
            throw new NotSupportedException($"Provider '{provider}' is not supported.");
        }

        if (!string.IsNullOrWhiteSpace(ortExtensionsPath))
        {
            options.RegisterCustomOpLibraryV2(ortExtensionsPath, out var libraryHandle);
        }

        options.EnableMemoryPattern = false;

        // TODO: Add configurable logging settings
        options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
        options.LogVerbosityLevel = 0; // Default = 0. Valid values are >= 0.This takes into effect only when the LogSeverityLevel is set to ORT_LOGGING_LEVEL_VERBOSE.

        return new InferenceSession(path, options);
    }

    /// <summary>
    /// Load a model from a directory
    /// </summary>
    /// <param name="modelId">Folder from which to load</param>
    /// <param name="revision">Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id</param>
    /// <param name="forceDownload">Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist.</param>
    /// <param name="useAuthToken">Is needed to load models from a private or gated repository</param>
    /// <param name="cacheDir">Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.</param>
    /// <param name="fileName">Overwrites the default model file name from <c>model.onnx</c> to <c>fileName</c>. This allows you to load different model files from the same repository or folder.</param>
    /// <param name="provider">The ONNX runtime provider, e.g. <c>CPUExecutionProvider</c> or <c>CUDAExecutionProvider</c>.</param>
    /// <param name="sessionOptions">The ONNX runtime provider options</param>
    /// <exception cref="NotImplementedException"></exception>
    /// <returns></returns>
    public static OnnxRuntimeModel FromPretrained(string modelId, string? revision = null, string? fileName = null, string? provider = null, Dictionary<string, string>? sessionOptions = null, string? ortExtensionPath = null)
    {
        var modelIdParts = modelId.Split('@');
        if (modelIdParts.Length == 2)
        {
            modelId = modelIdParts[0];
            revision = modelIdParts[1];
        }

        string modelFileName = fileName ?? DiffusionPipeline.ONNX_WEIGHTS_NAME;
        string? modelSaveDir = null;
        string? latestModelName = null;
        InferenceSession model;

        // Load model from local folder
        if (Directory.Exists(modelId))
        {
            model = LoadModel(Path.Join(modelId, modelFileName), provider: provider, sessionOptions: sessionOptions, ortExtensionPath);
            modelSaveDir = modelId;
        }
        else
        {
            throw new InvalidOperationException($"The specified {modelId} cannot be found");
        }

        return new OnnxRuntimeModel(model, modelSaveDir: modelSaveDir, latestModelName: latestModelName);
    }

    public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        return _model.Run(inputs);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                // Dispose managed state (managed objects)
                _model.Dispose();
            }

            // Free unmanaged resources (unmanaged objects) and override finalizer
            // Set large fields to null
            _disposed = true;
        }
    }

    // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
    // ~OnnxRuntimeModel()
    // {
    //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
    //     Dispose(disposing: false);
    // }

    public void Dispose()
    {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}