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

namespace GenAIPlayground.StableDiffusion.MLModels;

using GenAIPlayground.StableDiffusion.MLModels.Interfaces;

/// <summary>
/// Base class for all models.<br />
/// <c>DiffusionPipeline</c> takes care of storing all components (models, schedulers, processors) for diffusion pipelines.
/// </summary>
public abstract class DiffusionPipeline : IDiffusionPipeline
{
    public static readonly string MODEL_CONFIG_FILENAME = "model_index.json";
    public static readonly string ONNX_WEIGHTS_NAME = "model.onnx";

    public abstract StableDiffusionPipelineOutput Run(List<string> prompts, List<string> negativePrompts, StableDiffusionConfig config);
}