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

namespace SharpDiffusion;

using SharpDiffusion.Interfaces;

/// <summary>
/// Base class for all models.<br />
/// <c>DiffusionPipeline</c> takes care of storing all components (models, schedulers, processors) for diffusion pipelines.
/// </summary>
public abstract class DiffusionPipeline : IDiffusionPipeline, IDisposable
{
    public static readonly string MODEL_CONFIG_FILENAME = "model_index.json";
    public static readonly string ONNX_WEIGHTS_NAME = "model.onnx";
    protected bool _disposed;

    public abstract StableDiffusionPipelineOutput Run(List<string> prompts, List<string> negativePrompts, StableDiffusionConfig config, Action<int>? callback = null);

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                // TODO: dispose managed state (managed objects)
            }

            // TODO: free unmanaged resources (unmanaged objects) and override finalizer
            // TODO: set large fields to null
            _disposed = true;
        }
    }

    // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
    // ~DiffusionPipeline()
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